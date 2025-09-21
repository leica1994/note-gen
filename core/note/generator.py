from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from textwrap import dedent
from time import perf_counter
from typing import List, Dict, Any, Set, Iterable

from langchain_core.messages import SystemMessage, HumanMessage

from core.config.schema import AppConfig
from core.llms.mm_llm import MultiModalLLM
from core.llms.text_llm import TextLLM
from core.note.models import (
    Chapter,
    ChapterBoundary,
    ChaptersSchema,
    GenerationInputMeta,
    Note,
    Paragraph,
    ParagraphImage,
    ParagraphLine,
    ParagraphSchema,
    ParagraphsSchema,
)
from core.screenshot.ffmpeg import Screenshotter
from core.screenshot.grid import generate_grid_timestamps
from core.subtitles.models import SubtitleSegment


class NoteGenerator:
    """核心笔记生成器。

    流程：
    1) 分章：基于全部字幕（行号+时间）让 LLM 输出章节边界
    2) 分段：逐章节调用 LLM，生成层次段落（保留全部字幕行；时间覆盖完整、无重叠）
    3) 选图：对每个段落等距生成 9 帧缩略图 → 九宫格 → 多模态 LLM 选择最佳索引
    4) 重拍：按被选中的时间戳重拍高清单帧
    5) 汇总：生成 Note 对象，用于后续导出 Markdown
    """

    def __init__(self, config: AppConfig, logger: logging.Logger | None = None) -> None:
        self.cfg = config
        self.text_llm = TextLLM(config.text_llm)
        self.mm_llm = MultiModalLLM(config.mm_llm)
        self.screenshotter = Screenshotter(config.screenshot)
        self.logger = logger or logging.getLogger("note_gen.generator")
        # 并发与限流：
        self._text_sem = threading.Semaphore(max(1, self.cfg.text_llm.concurrency))
        self._mm_sem = threading.Semaphore(max(1, self.cfg.mm_llm.concurrency))
        # 全局截图并发：
        # - 九宫格缩略图：受全局信号量限制（允许若干并发，降低排队放大）
        # - 高清重拍：保持全局串行，避免 I/O 抖动并保证稳定性
        self._thumb_sem = threading.Semaphore(max(1, self.cfg.screenshot.max_workers))
        self._shot_sem = threading.Semaphore(1)

    def _render_all_subtitles(self, meta: GenerationInputMeta) -> str:
        lines = []
        for seg in meta.subtitle.items:
            lines.append(f"{seg.line_no}\t[{seg.start_sec:.3f}-{seg.end_sec:.3f}]\t{seg.text}")
        return "\n".join(lines)

    def _prompt_for_chapters(self, meta: GenerationInputMeta) -> ChaptersSchema:
        sys = SystemMessage(content=dedent(
            """
            你是一名专业的结构化编辑，任务是将有序字幕拆分为多个'章节'。
            
            严格要求：
            - 按字幕行号连续覆盖，禁止缺失或重叠；
            - 每个章节返回：title, start_line_no, end_line_no；
            - 章节按行号升序排列；
            - 不做任何摘要或改写，不丢失行；
            - 章节标题必须为简体中文表达，避免英文/拼音/纯时间/纯行号；如模型拟输出英文标题，请翻译为准确的中文短语。
            
            章节划分原则：
            - 仅在出现明显“主题切换”时新开章节：如引言→主体、主题/场景/对象变化、明确的枚举段落（第一/第二/…）、问答/讲解模式切换、长停顿（>1.5s）、时间/地点/人物切换等。
            - 章节应能以“名词短语/主题短句”命名，标题需准确概括该章的核心主题，避免纯时间/行号。
            - 避免将同一主题拆成多个相邻章节；若相邻片段围绕同一主题，应合并为一章。
            - 若某候选章节过短（例如仅涵盖极少字幕行或极短时长）且无明显边界信号，应与上下文合并，宁可更凝练也不要碎片化。
            """
        ))
        human = HumanMessage(content=dedent(
            f"""
            以下是完整字幕（包含行号与时间戳）。请给出章节边界：

            {self._render_all_subtitles(meta)}
            """
        ))
        t0 = perf_counter()
        result = self.text_llm.structured_invoke(ChaptersSchema, [sys, human], json_mode=False)

        # 标题后置中文化校验：若极端情况下仍出现非中文标题，以“第N章”占位保证中文化
        def _has_cjk(s: str) -> bool:
            return any('\u4e00' <= ch <= '\u9fff' for ch in s)

        for idx, ch in enumerate(result.chapters, start=1):
            if not ch.title or not _has_cjk(ch.title):
                ch.title = f"第{idx}章"
        self.logger.info("分章完成",
                         extra={"chapters": len(result.chapters), "cost_ms": int((perf_counter() - t0) * 1000)})
        # 证据
        # 调试证据归档能力已移除（简化发布版）
        return result

    def _select_lines(self, items: List[SubtitleSegment], start_line: int, end_line: int) -> List[SubtitleSegment]:
        return [s for s in items if start_line <= s.line_no <= end_line]

    def _prompt_for_paragraphs(self, chapter_title: str, segs: List[SubtitleSegment], fix_note: str | None = None,
                               prev_result: ParagraphsSchema | None = None) -> ParagraphsSchema:
        sys = SystemMessage(content=dedent(
            """
            你是一名专业的结构化编辑，任务是将给定章节内的字幕行分级成段落。
            严格要求：
            - 覆盖所有提供的字幕行；
            - 返回 paragraphs: 每个段落至少包含 title, start_sec, end_sec, lines；
            - lines 数组逐条包含 line_no, start_sec, end_sec, text（保持原文）；
            - 段落时间戳覆盖必须完整且不重叠，不得缺失；可以有子段落；
            - 不做摘要或改写；
            - 若没有子段落，children 必须返回空数组 []，不要返回空对象 {} 或 null；
            - 只能使用下方提供的行号，每个行号必须且仅出现一次；不得遗漏、不得重复、不得引入列表外的行号。
            - 同级段落按时间升序排列，互不重叠，顶层段落共同完整覆盖章节时间范围。

            额外输出：optimized
            - 为每个（含子层级）段落生成一个 optimized 字段：类型为字符串列表 List[str]；
            - 将该段落内所有字幕行去除语气词后，添加合适标点，拼接成流畅的句子；
            - 如果内容较长、可自然分段，请将 optimized 拆分为多条字符串；否则可只返回 1 条；
            - 对句子中的重点知识点请使用 Markdown 进行标记（例如 **加粗**、`行内代码`、列表等）；
            - 严禁在 optimized 中捏造未出现于该段落字幕中的事实；
            - 保留术语与数字的准确性，保持时序逻辑。
            """
        ))
        content_lines = "\n".join(
            f"{s.line_no}\t[{s.start_sec:.3f}-{s.end_sec:.3f}]\t{s.text}" for s in segs
        )
        fix_extra = f"\n\n修正提示：{fix_note}" if fix_note else ""
        prev_block = ""
        if prev_result is not None:
            # 上一轮失败的段落 JSON，用于指导本轮修正（最小变更）
            import json as _json
            prev_json = _json.dumps(prev_result.model_dump(), ensure_ascii=False, indent=2)
            prev_block = f"\n\n上一轮输出（供修正，JSON）：\n{prev_json}"
        human = HumanMessage(content=dedent(
            f"""
            章节标题：{chapter_title}
            请对以下字幕行进行结构化分段：

            {content_lines}

            注意：严格使用以上行号，确保完整覆盖且不重叠。{fix_extra}
            请在上一轮输出的基础上进行最小必要修改：补齐缺失行、移除重复/多余行，保持行号与时间戳与输入一致，不要改写文本，不要引入列表外行号。{prev_block}
            """
        ))
        with self._text_sem:
            result = self.text_llm.structured_invoke(ParagraphsSchema, [sys, human], json_mode=False)
        return result

    def _validate_time_coverage(self, segs: List[SubtitleSegment], paras: List[ParagraphSchema]) -> None:
        """校验：
        - 段落时间范围不重叠；
        - 段落整体覆盖章节时间范围；
        - 每段落 lines 时间均在其段落范围内；
        - 段落 lines 的行号并集等于输入 segs 的行号集合（无缺失/重复）。
        """
        if not paras:
            raise ValueError("分段结果为空")
        ranges = sorted([(p.start_sec, p.end_sec) for p in paras], key=lambda x: x[0])
        for i in range(1, len(ranges)):
            if ranges[i][0] < ranges[i - 1][1]:
                raise ValueError("分段时间范围存在重叠")
        min_in = min(s.start_sec for s in segs)
        max_in = max(s.end_sec for s in segs)
        if abs(ranges[0][0] - min_in) > 0.05 or abs(ranges[-1][1] - max_in) > 0.05:
            raise ValueError("分段时间范围未完整覆盖章节区间")

        # 行号覆盖检查（递归）
        def iter_lines(ps: List[ParagraphSchema]):
            for p in ps:
                for l in p.lines:
                    yield l
                if p.children:
                    yield from iter_lines(p.children)

        input_ids = [s.line_no for s in segs]
        out_ids = [l.line_no for l in iter_lines(paras)]
        if sorted(input_ids) != sorted(out_ids):
            raise ValueError("分段行号集合与输入集合不一致，存在缺失或重复")

        # 每段落内行时间应在段落区间内，且段落边界应等于该段落所有行的最小开始/最大结束
        for p in paras:
            for l in p.lines:
                if not (p.start_sec - 0.05 <= l.start_sec <= l.end_sec <= p.end_sec + 0.05):
                    raise ValueError("段落内行时间越界")
            min_line_start = min(l.start_sec for l in p.lines)
            max_line_end = max(l.end_sec for l in p.lines)
            if abs(p.start_sec - min_line_start) > 0.05 or abs(p.end_sec - max_line_end) > 0.05:
                raise ValueError("段落边界应等于该段落所有行的最小开始与最大结束")

    def _analyze_coverage_issue(self, segs: List[SubtitleSegment], paras: List[ParagraphSchema]) -> Dict[str, Any]:
        """分析段落覆盖问题（缺失/多余/重复/重叠/覆盖缺口）。"""
        expect: Set[int] = {s.line_no for s in segs}

        def flatten_lines(ps: List[ParagraphSchema]) -> Iterable[int]:
            for p in ps:
                for l in p.lines:
                    yield l.line_no
                if p.children:
                    yield from flatten_lines(p.children)

        got_list = list(flatten_lines(paras))
        got: Set[int] = set(got_list)
        duplicate: Set[int] = {x for x in got_list if got_list.count(x) > 1}
        missing = sorted(list(expect - got))
        extra = sorted(list(got - expect))

        ranges = sorted([(p.start_sec, p.end_sec) for p in paras], key=lambda x: x[0])
        overlap = any(ranges[i][0] < ranges[i - 1][1] for i in range(1, len(ranges))) if len(ranges) > 1 else False
        min_in = min(s.start_sec for s in segs)
        max_in = max(s.end_sec for s in segs)
        coverage_ok = bool(ranges) and (abs(ranges[0][0] - min_in) <= 0.05 and abs(ranges[-1][1] - max_in) <= 0.05)

        # 段落边界与内部行时间不一致的统计
        out_of_bounds = 0
        for p in paras:
            min_line_start = min(l.start_sec for l in p.lines) if p.lines else p.start_sec
            max_line_end = max(l.end_sec for l in p.lines) if p.lines else p.end_sec
            if not (abs(p.start_sec - min_line_start) <= 0.05 and abs(p.end_sec - max_line_end) <= 0.05):
                out_of_bounds += 1

        return {
            "missing": missing,
            "extra": extra,
            "duplicate": sorted(list(duplicate)),
            "overlap": overlap,
            "coverage_ok": coverage_ok,
            "boundary_mismatch": out_of_bounds,
        }

    def _auto_fix_paragraph_lines(
            self,
            segs: List[SubtitleSegment],
            paragraphs: List[ParagraphSchema],
            issue: Dict[str, Any] | None = None,
    ) -> Dict[str, int]:
        """在重试耗尽后自动修正段落的字幕行集合，使其与输入字幕一致。

        策略：
        1) 去重/剔除额外行 → 仅保留来自 segs 的合法行且唯一；
        2) 补齐缺失行 → 按时间/行号接近度分配到最合适段落（或新建段落）；
        3) 规范化：
           - 递归清洗空节点，行按行号排序，段落边界 = 行最小开始/最大结束；
           - 顶层段落按最小行号排序，若时间范围重叠则合并（行集合并、边界随之更新）。
        4) 校验前强制兜底：若仍与输入集合不一致，则“扁平化重建”为单一顶层段落，确保通过集合与覆盖校验（牺牲结构保可靠）。
        5) 返回修正统计供审计。
        """
        line_map: Dict[int, SubtitleSegment] = {s.line_no: s for s in segs}
        expected_ids: Set[int] = set(line_map.keys())
        assigned: Set[int] = set()
        stats: Dict[str, int] = {
            "removed_duplicate": 0,
            "removed_extra": 0,
            "filled_missing": 0,
            "created_paragraphs": 0,
            "promoted_children": 0,
            "removed_empty": 0,
            "merged_overlaps": 0,
            "reconstructed_flatten": 0,
            "final_missing": 0,
            "final_extra": 0,
        }
        tolerance = 0.3

        def iter_nodes(nodes: List[ParagraphSchema]) -> Iterable[ParagraphSchema]:
            for node in nodes:
                yield node
                if node.children:
                    yield from iter_nodes(node.children)

        def collect_ids(nodes: List[ParagraphSchema]) -> Iterable[int]:
            for node in nodes:
                for line in node.lines:
                    yield line.line_no
                if node.children:
                    yield from collect_ids(node.children)

        def deduplicate(nodes: List[ParagraphSchema]) -> None:
            for node in nodes:
                new_lines: List[ParagraphLine] = []
                for line in node.lines:
                    seg = line_map.get(line.line_no)
                    if seg is None:
                        stats["removed_extra"] += 1
                        continue
                    if line.line_no in assigned:
                        stats["removed_duplicate"] += 1
                        continue
                    assigned.add(line.line_no)
                    new_lines.append(
                        ParagraphLine(
                            line_no=seg.line_no,
                            start_sec=seg.start_sec,
                            end_sec=seg.end_sec,
                            text=seg.text,
                        )
                    )
                node.lines = new_lines
                if node.children:
                    deduplicate(node.children)

        deduplicate(paragraphs)

        missing_ids = sorted(expected_ids - assigned)
        for line_no in missing_ids:
            seg = line_map[line_no]

            def pick_target() -> ParagraphSchema | None:
                best: tuple[float, ParagraphSchema] | None = None
                for node in iter_nodes(paragraphs):
                    if seg.start_sec >= node.start_sec - tolerance and seg.end_sec <= node.end_sec + tolerance:
                        gap = abs(seg.start_sec - node.start_sec) + abs(seg.end_sec - node.end_sec)
                        if best is None or gap < best[0]:
                            best = (gap, node)
                if best is not None:
                    return best[1]

                best = None
                for node in iter_nodes(paragraphs):
                    if node.lines:
                        min_line = node.lines[0].line_no
                        max_line = node.lines[-1].line_no
                        if min_line <= seg.line_no <= max_line:
                            return node
                        gap = min(abs(seg.line_no - min_line), abs(seg.line_no - max_line))
                    else:
                        gap = abs(seg.start_sec - node.start_sec)
                    if best is None or gap < best[0]:
                        best = (gap, node)
                return best[1] if best else None

            target = pick_target()
            new_line = ParagraphLine(
                line_no=seg.line_no,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                text=seg.text,
            )
            stats["filled_missing"] += 1
            assigned.add(seg.line_no)
            if target is not None:
                target.lines.append(new_line)
            else:
                paragraphs.append(
                    ParagraphSchema(
                        title=f"自动补齐 {seg.line_no}",
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        lines=[new_line],
                    )
                )
                stats["created_paragraphs"] += 1

        def finalize(nodes: List[ParagraphSchema]) -> List[ParagraphSchema]:
            result: List[ParagraphSchema] = []
            for node in nodes:
                node.children = finalize(node.children)
                if node.lines:
                    node.lines.sort(key=lambda x: x.line_no)
                    node.start_sec = min(line.start_sec for line in node.lines)
                    node.end_sec = max(line.end_sec for line in node.lines)
                    result.append(node)
                elif node.children:
                    result.extend(node.children)
                    node.children = []
                    stats["promoted_children"] += 1
                else:
                    stats["removed_empty"] += 1
            return result

        paragraphs[:] = finalize(paragraphs)

        # 顶层段落排序并合并重叠时间范围，保证校验通过（不改变行集合）
        def _para_min_line_no(p: ParagraphSchema) -> int:
            return min((l.line_no for l in p.lines), default=10 ** 9)

        paragraphs.sort(key=_para_min_line_no)
        merged: List[ParagraphSchema] = []
        for p in paragraphs:
            if not merged:
                merged.append(p)
                continue
            prev = merged[-1]
            if p.start_sec < prev.end_sec:
                # 合并到上一段
                prev.lines.extend(p.lines)
                prev.lines.sort(key=lambda x: x.line_no)
                prev.start_sec = min(line.start_sec for line in prev.lines)
                prev.end_sec = max(line.end_sec for line in prev.lines)
                prev.children = []
                stats["merged_overlaps"] += 1
            else:
                merged.append(p)
        paragraphs[:] = merged

        final_ids = set(collect_ids(paragraphs))
        if final_ids != expected_ids:
            # 兜底：扁平化重建为单一顶层段落，确保集合一致且覆盖完整
            new_lines = [
                ParagraphLine(
                    line_no=s.line_no,
                    start_sec=s.start_sec,
                    end_sec=s.end_sec,
                    text=s.text,
                )
                for s in segs
            ]
            if new_lines:
                paragraphs[:] = [
                    ParagraphSchema(
                        title="自动合并",
                        start_sec=min(s.start_sec for s in segs),
                        end_sec=max(s.end_sec for s in segs),
                        lines=new_lines,
                        children=[],
                    )
                ]
                stats["reconstructed_flatten"] = 1
                final_ids = set(collect_ids(paragraphs))

        stats["final_missing"] = len(expected_ids - final_ids)
        stats["final_extra"] = len(final_ids - expected_ids)

        if issue is not None:
            # 保留原始异常信息，协助定位LLM输出习惯性问题
            stats.setdefault("issue_missing", len(issue.get("missing", [])))
            stats.setdefault("issue_duplicate", len(issue.get("duplicate", [])))
            stats.setdefault("issue_extra", len(issue.get("extra", [])))

        return stats

    def _choose_best_frame(self, para: Paragraph, grid_path: Path, ci: int, pi: int, subpath: str | None = None) -> int:
        text = "\n".join(f"{s.line_no}: {s.text}" for s in para.lines)
        instruction = dedent(
            f"""
            任务：从九宫格（3x3，共9张）中选择最能表达本段内容的一张截图。

            强制拒绝以下类型（视为过渡帧）：
            - 交叉淡入淡出/叠化效果，画面中同时出现前后两帧元素（重影、双层文字、两张图重叠）。
            - 明显运动模糊、拖影、未对焦、被切换动画覆盖的帧。
            - 文字或图表被截断、被大块字幕/弹窗/菜单遮挡、窗口切换中的半透明覆盖层。

            选择偏好：
            - 画面清晰锐利、对比正常、主题居中完整；
            - 若有K线/图表，尽量选择结构完整、无裁切、无遮挡的一张；
            - 若多张都合格，优先选择与段落标题和字幕语义最贴合的一张；
            - 若仍难以区分，选择信息量更高、元素更稳定的一张。

            输出规范：仅返回一个阿拉伯数字（1-9），不含空格与其他字符。无法判断时返回 5。

            段落标题：{para.title}
            段落内容：
            {text}
            """
        )
        # 受多模态 LLM 并发限制
        with self._mm_sem:
            chosen = self.mm_llm.choose_index(instruction, str(grid_path))
        # 证据归档移除
        return int(chosen)

    def _convert_paragraph(self, ps: ParagraphSchema) -> Paragraph:
        lines = [
            SubtitleSegment(line_no=l.line_no, start_sec=l.start_sec, end_sec=l.end_sec, text=l.text)
            for l in ps.lines
        ]
        children = [self._convert_paragraph(c) for c in ps.children] if ps.children else []
        return Paragraph(
            title=ps.title,
            start_sec=ps.start_sec,
            end_sec=ps.end_sec,
            lines=lines,
            children=children,
            optimized=list(ps.optimized or []),
        )

    def _process_chapter(self, ci: int, cb: ChapterBoundary, meta: GenerationInputMeta, task_out_dir: Path) -> Chapter:
        self.logger.info("处理章节", extra={"chapter_index": ci, "title": cb.title})
        segs = self._select_lines(meta.subtitle.items, cb.start_line_no, cb.end_line_no)
        if not segs:
            raise ValueError(f"章节无内容：{cb}")

        # 分段（业务重试，最多3次）
        max_attempts = 3
        last_issue: Dict[str, Any] | None = None
        last_result: ParagraphsSchema | None = None
        pgs: ParagraphsSchema | None = None
        for attempt in range(1, max_attempts + 1):
            fix_note = None
            if last_issue:
                parts = []
                if last_issue.get("missing"):
                    parts.append(f"缺失行号: {last_issue['missing']}")
                if last_issue.get("extra"):
                    parts.append(f"多余行号: {last_issue['extra']}")
                if last_issue.get("duplicate"):
                    parts.append(f"重复行号: {last_issue['duplicate']}")
                if last_issue.get("overlap"):
                    parts.append("段落时间重叠")
                if not last_issue.get("coverage_ok", True):
                    parts.append("时间未完整覆盖章节范围")
                if last_issue.get("boundary_mismatch"):
                    parts.append("段落边界必须等于该段落所有行的最小开始与最大结束；禁止修改任何行的时间戳")
                fix_note = "；".join(parts)

            pgs = self._prompt_for_paragraphs(cb.title, segs, fix_note=fix_note, prev_result=last_result)

            # 证据归档能力已移除（简化发布版）

            try:
                self._validate_time_coverage(segs, pgs.paragraphs)
                self.logger.info("章节校验通过",
                                 extra={"chapter_index": ci, "paragraphs": len(pgs.paragraphs), "attempt": attempt})
                break
            except Exception:
                # 记录当前失败输出，供下一轮提示参考
                last_result = pgs
                issue = self._analyze_coverage_issue(segs, pgs.paragraphs)
                last_issue = issue
                self.logger.info("章节校验失败，准备重试", extra={"chapter_index": ci, "attempt": attempt, **issue})
                if attempt >= max_attempts:
                    self.logger.warning(
                        "章节校验连续失败，触发自动修正",
                        extra={"chapter_index": ci, "attempt": attempt, **issue},
                    )
                    try:
                        fix_report = self._auto_fix_paragraph_lines(segs, pgs.paragraphs, issue)
                        self._validate_time_coverage(segs, pgs.paragraphs)
                        self.logger.info(
                            "自动修正完成",
                            extra={
                                "chapter_index": ci,
                                "attempt": attempt,
                                "auto_fix": fix_report,
                            },
                        )
                        break
                    except Exception as fix_error:
                        self.logger.error(
                            "自动修正失败",
                            extra={"chapter_index": ci, "attempt": attempt, "error": str(fix_error)},
                        )
                        raise ValueError("分段行号集合与输入集合不一致，存在缺失或重复") from fix_error

        assert pgs is not None

        # 段落并发：截图→九宫格→多模态→高清
        def _decorate_with_images(para: Paragraph, base_dir: Path, label: str | None = None,
                                  para_index: int | None = None) -> None:
            # 生成九宫格并选择
            timestamps = generate_grid_timestamps(para.start_sec, para.end_sec, self.cfg.screenshot)
            thumbs_dir = base_dir / "thumbs"
            grid_path = base_dir / "grid.jpg"
            t_s = perf_counter()
            try:
                with self._thumb_sem:
                    self.screenshotter.compose_grid_one_shot(
                        Path(meta.video_path), timestamps, grid_path,
                        cols=self.cfg.screenshot.grid_columns,
                        rows=self.cfg.screenshot.grid_rows,
                        width=self.cfg.screenshot.low_width,
                        height=self.cfg.screenshot.low_height,
                    )
            except Exception as e:
                # 回退方案：逐帧缩略图 + PIL 拼图
                self.logger.info("一把流九宫格失败，将回退逐帧", extra={
                    "chapter_index": ci,
                    "para_index": para_index or 0,
                    "sub_index": label or "",
                    "error": str(e),
                })
                with self._thumb_sem:
                    thumbs = self.screenshotter.capture_thumbs(Path(meta.video_path), timestamps, thumbs_dir)
                    self.screenshotter.compose_grid(thumbs, grid_path)
            self.logger.info("生成九宫格完成", extra={
                "chapter_index": ci,
                "para_index": para_index or 0,
                "sub_index": label or "",
                "grid": str(grid_path),
                "cost_ms": int((perf_counter() - t_s) * 1000),
            })
            chosen_idx = self._choose_best_frame(para, grid_path, ci, para_index or 0, subpath=label)
            chosen_ts = timestamps[chosen_idx - 1]
            # 高清重拍
            # 高清截图文件名：视频名称_段落标题_时间戳（去除符号），如 11:11:11 -> 111111
            def _sanitize_filename(name: str) -> str:
                """将任意字符串转换为安全文件名：移除非法字符并将空白替换为下划线。"""
                import re
                # 去除路径分隔与常见非法字符
                name = re.sub(r"[\\/\\?%\*:|\"<>]", "", name)
                # 替换空白为下划线，收尾裁剪
                name = re.sub(r"\s+", "_", name).strip("._ ")
                # 控制长度，过长时截断
                return name[:150] or "untitled"

            def _format_ts_hhmmss(ts: float) -> str:
                t = max(0, int(ts))
                hh = t // 3600
                mm = (t % 3600) // 60
                ss = t % 60
                return f"{hh:02d}{mm:02d}{ss:02d}"

            video_stem = Path(meta.video_path).stem
            ts_str = _format_ts_hhmmss(chosen_ts)
            safe_title = _sanitize_filename(para.title or "")
            safe_video = _sanitize_filename(video_stem)
            hi_name = f"{safe_video}_{safe_title}_{ts_str}.jpg"
            # 高清图输出位置：
            # - 若 GUI 配置了截图目录（note.screenshot_input_dir），则写入该目录；
            # - 否则仍写入任务输出目录（base_dir）。
            hi_root = None
            try:
                hi_root = getattr(getattr(self.cfg, 'note', None), 'screenshot_input_dir', None)
            except Exception:
                hi_root = None
            if hi_root:
                out_root = Path(hi_root)
            else:
                out_root = base_dir
            try:
                out_root.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            hi_path = out_root / hi_name
            t_hq = perf_counter()
            with self._shot_sem:
                self.screenshotter.capture_high_quality(Path(meta.video_path), chosen_ts, hi_path)
            self.logger.info("高清重拍完成", extra={
                "chapter_index": ci,
                "para_index": para_index or 0,
                "sub_index": label or "",
                "chosen_index": chosen_idx,
                "timestamp_sec": chosen_ts,
                "hi_image": str(hi_path),
                "cost_ms": int((perf_counter() - t_hq) * 1000),
            })
            para.image = ParagraphImage(
                grid_image_path=grid_path,
                grid_timestamps_sec=timestamps,
                chosen_index=chosen_idx,
                chosen_timestamp_sec=chosen_ts,
                hi_res_image_path=hi_path,
            )

            # 递归处理子段落（在父段落任务内顺序执行，避免爆炸并发）
            if para.children:
                for si, child in enumerate(para.children, start=1):
                    child_dir = base_dir / f"sub_{si}"
                    sub_label = f"{label}/sub_{si}" if label else f"sub_{si}"
                    _decorate_with_images(child, child_dir, sub_label, para_index=para_index)

        def _process_para(pi: int, ps: ParagraphSchema) -> Paragraph:
            # 递归转换
            para = self._convert_paragraph(ps)
            base_dir = task_out_dir / f"chapter_{ci}" / f"para_{pi}"
            _decorate_with_images(para, base_dir, None, para_index=pi)
            return para

        paragraphs: List[Paragraph] = [None] * len(pgs.paragraphs)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.screenshot.max_workers)) as ex:
            future_map = {}
            for idx, ps in enumerate(pgs.paragraphs, start=1):
                fut = ex.submit(_process_para, idx, ps)
                future_map[fut] = idx
            for fut in as_completed(future_map):
                idx = future_map[fut]
                paragraphs[idx - 1] = fut.result()

        chapter = Chapter(
            title=cb.title,
            start_sec=min(s.start_sec for s in segs),
            end_sec=max(s.end_sec for s in segs),
            paragraphs=paragraphs,  # type: ignore[arg-type]
        )
        return chapter

    def generate(self, meta: GenerationInputMeta, task_out_dir: Path) -> Note:
        # 1) 分章
        chs = self._prompt_for_chapters(meta)
        # 发布版不再归档证据

        # 2) 章节并发处理
        chapters: List[Chapter] = [None] * len(chs.chapters)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.text_llm.concurrency)) as ex:
            future_map = {}
            for ci, cb in enumerate(chs.chapters, start=1):
                fut = ex.submit(self._process_chapter, ci, cb, meta, task_out_dir)
                future_map[fut] = ci
            for fut in as_completed(future_map):
                ci = future_map[fut]
                chapters[ci - 1] = fut.result()

        note = Note(
            video_path=Path(meta.video_path),
            subtitle=meta.subtitle,
            chapters=chapters,
            meta={
                "text_llm_model": self.cfg.text_llm.model,
                "mm_llm_model": self.cfg.mm_llm.model,
            },
        )
        self.logger.info("组装 Note 完成", extra={"chapters": len(note.chapters)})
        return note
