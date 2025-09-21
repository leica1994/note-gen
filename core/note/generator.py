from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import List, Dict, Any, Set, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from langchain_core.messages import SystemMessage, HumanMessage
from textwrap import dedent

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
    ParagraphSchema,
    ParagraphsSchema,
)
from core.screenshot.ffmpeg import Screenshotter
from core.screenshot.grid import generate_grid_timestamps
from core.subtitles.models import SubtitleSegment
from core.utils.evidence import EvidenceWriter
import logging


class NoteGenerator:
    """核心笔记生成器。

    流程：
    1) 分章：基于全部字幕（行号+时间）让 LLM 输出章节边界
    2) 分段：逐章节调用 LLM，生成层次段落（保留全部字幕行；时间覆盖完整、无重叠）
    3) 选图：对每个段落等距生成 9 帧缩略图 → 九宫格 → 多模态 LLM 选择最佳索引
    4) 重拍：按被选中的时间戳重拍高清单帧
    5) 汇总：生成 Note 对象，用于后续导出 Markdown
    """

    def __init__(self, config: AppConfig, evidence: EvidenceWriter, logger: logging.Logger | None = None) -> None:
        self.cfg = config
        self.evidence = evidence
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
            - 不做任何摘要或改写，不丢失行。
            
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
        self.logger.info("分章完成", extra={"chapters": len(result.chapters), "cost_ms": int((perf_counter()-t0)*1000)})
        # 证据
        if self.cfg.export.save_prompts_and_raw:
            self.evidence.write_text("chapters_prompt.txt", f"{sys.content}\n\n{human.content}")
            self.evidence.write_json("chapters_result.json", result.model_dump())
        return result

    def _select_lines(self, items: List[SubtitleSegment], start_line: int, end_line: int) -> List[SubtitleSegment]:
        return [s for s in items if start_line <= s.line_no <= end_line]

    def _prompt_for_paragraphs(self, chapter_title: str, segs: List[SubtitleSegment], fix_note: str | None = None) -> ParagraphsSchema:
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
        human = HumanMessage(content=dedent(
            f"""
            章节标题：{chapter_title}
            请对以下字幕行进行结构化分段：

            {content_lines}

            注意：严格使用以上行号，确保完整覆盖且不重叠。{fix_extra}
            """
        ))
        t0 = perf_counter()
        # 受文本 LLM 并发限制
        with self._text_sem:
            result = self.text_llm.structured_invoke(ParagraphsSchema, [sys, human], json_mode=False)
        self.logger.info("分段完成", extra={"paragraphs": len(result.paragraphs), "cost_ms": int((perf_counter()-t0)*1000)})
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

    def _choose_best_frame(self, para: Paragraph, grid_path: Path, ci: int, pi: int) -> int:
        text = "\n".join(f"{s.line_no}: {s.text}" for s in para.lines)
        instruction = dedent(
            f"""
            请从下方九宫格图中选择与该段落标题与内容最匹配的一张截图。
            要求：避免过渡帧，画面清晰稳定。
            仅返回一个数字（1-9），不要任何其他字符、空格或换行；无法判断时返回 5。
            段落标题：{para.title}
            段落内容：
            {text}
            """
        )
        # 受多模态 LLM 并发限制
        with self._mm_sem:
            chosen = self.mm_llm.choose_index(instruction, str(grid_path))
        if self.cfg.export.save_prompts_and_raw:
            self.evidence.write_text(f"chapters/{ci}/para_{pi}/choose_image_instruction.txt", instruction)
            self.evidence.write_text(f"chapters/{ci}/para_{pi}/choose_image_result.txt", str(chosen))
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
                    parts.append("段落边界必须等于该段落所有行的最小开始与最大结束；禁止修改任一行的时间戳")
                fix_note = "；".join(parts)

            pgs = self._prompt_for_paragraphs(cb.title, segs, fix_note=fix_note)

            # 证据：按尝试次数归档
            if self.cfg.export.save_prompts_and_raw:
                content_lines = "\n".join(
                    f"{s.line_no}\t[{s.start_sec:.3f}-{s.end_sec:.3f}]\t{s.text}" for s in segs
                )
                fix_extra = f"\n\n修正提示：{fix_note}" if fix_note else ""
                prompt_text = dedent(
                    f"""
                    你是一名专业的结构化编辑，任务是将给定章节内的字幕行分级成段落。
                    严格要求：
                    - 覆盖所有提供的字幕行；
                    - 返回 paragraphs: 每个段落至少包含 title, start_sec, end_sec, lines；
                    - lines 数组逐条包含 line_no, start_sec, end_sec, text（保持原文）；
                    - 段落时间戳覆盖必须完整且不重叠，不得缺失；可以有子段落；
                    - 不做摘要或改写；
                    - 若没有子段落，children 必须返回空数组 []，不要返回空对象 {{}} 或 null；
                    - 只能使用下方提供的行号，每个行号必须且仅出现一次；不得遗漏、不得重复、不得引入列表外的行号。
                    - 同级段落按时间升序排列，互不重叠，顶层段落共同完整覆盖章节时间范围。

                    额外输出：optimized
                    - 为每个（含子层级）段落生成一个 optimized 字段：类型为字符串列表 List[str]；
                    - 将该段落内所有字幕行去除语气词后，添加合适标点，拼接成流畅的句子；
                    - 如果内容较长、可自然分段，请将 optimized 拆分为多条字符串；否则可只返回 1 条；
                    - 对句子中的重点知识点请使用 Markdown 进行标记（例如 **加粗**、`行内代码`、列表等）；
                    - 严禁在 optimized 中捏造未出现于该段落字幕中的事实；
                    - 保留术语与数字的准确性，保持时序逻辑。

                    章节标题：{cb.title}
                    {content_lines}
                    {fix_extra}
                    """
                )
                self.evidence.write_text(f"chapters/{ci}/attempt_{attempt}/paragraphs_prompt.txt", prompt_text)
                self.evidence.write_json(f"chapters/{ci}/attempt_{attempt}/paragraphs_result.json", pgs.model_dump())

            try:
                self._validate_time_coverage(segs, pgs.paragraphs)
                self.logger.info("章节校验通过", extra={"chapter_index": ci, "paragraphs": len(pgs.paragraphs), "attempt": attempt})
                break
            except Exception:
                issue = self._analyze_coverage_issue(segs, pgs.paragraphs)
                last_issue = issue
                self.logger.info("章节校验失败，准备重试", extra={"chapter_index": ci, "attempt": attempt, **issue})
                if attempt >= max_attempts:
                    raise ValueError("分段行号集合与输入集合不一致，存在缺失或重复")

        assert pgs is not None

        # 段落并发：截图→九宫格→多模态→高清
        def _process_para(pi: int, ps: ParagraphSchema) -> Paragraph:
            # 递归转换
            para = self._convert_paragraph(ps)
            # 生成九宫格并选择
            timestamps = generate_grid_timestamps(para.start_sec, para.end_sec, self.cfg.screenshot)
            thumbs_dir = task_out_dir / f"chapter_{ci}" / f"para_{pi}" / "thumbs"
            grid_path = task_out_dir / f"chapter_{ci}" / f"para_{pi}" / "grid.jpg"
            t_s = perf_counter()
            # 优先使用“一次 ffmpeg 产出九宫格”（失败时回退到旧方案）
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
                    "para_index": pi,
                    "error": str(e),
                })
                with self._thumb_sem:
                    thumbs = self.screenshotter.capture_thumbs(Path(meta.video_path), timestamps, thumbs_dir)
                    self.screenshotter.compose_grid(thumbs, grid_path)
            self.logger.info("生成九宫格完成", extra={
                "chapter_index": ci,
                "para_index": pi,
                "grid": str(grid_path),
                "cost_ms": int((perf_counter()-t_s)*1000),
            })
            chosen_idx = self._choose_best_frame(para, grid_path, ci, pi)
            chosen_ts = timestamps[chosen_idx - 1]
            # 高清重拍
            hi_path = task_out_dir / f"chapter_{ci}" / f"para_{pi}" / "hi.jpg"
            t_hq = perf_counter()
            with self._shot_sem:
                self.screenshotter.capture_high_quality(Path(meta.video_path), chosen_ts, hi_path)
            self.logger.info("高清重拍完成", extra={
                "chapter_index": ci,
                "para_index": pi,
                "chosen_index": chosen_idx,
                "timestamp_sec": chosen_ts,
                "hi_image": str(hi_path),
                "cost_ms": int((perf_counter()-t_hq)*1000),
            })
            para.image = ParagraphImage(
                grid_image_path=grid_path,
                grid_timestamps_sec=timestamps,
                chosen_index=chosen_idx,
                chosen_timestamp_sec=chosen_ts,
                hi_res_image_path=hi_path,
            )
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
        if self.cfg.export.save_prompts_and_raw:
            # 保存分章证据已在 _prompt_for_chapters 中记录，此处不重复
            pass

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
