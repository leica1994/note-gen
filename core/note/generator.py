from __future__ import annotations

import logging
import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from textwrap import dedent
from time import perf_counter
from typing import Any, Dict, Iterable, List, Set, Tuple

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
    ParagraphSchema,
    ParagraphsSchema,
)
from core.screenshot.ffmpeg import Screenshotter
from core.screenshot.grid import generate_grid_timestamps
from core.subtitles.models import SubtitleDocument, SubtitleSegment


from dataclasses import dataclass, field


@dataclass
class ParagraphRenderTask:
    chapter_index: int
    chapter_path: str
    chapter_title: str
    paragraph_index: int
    paragraph: Paragraph
    base_dir: Path
    label: str | None = None
    timestamps: List[float] = field(default_factory=list)
    grid_path: Path | None = None
    chosen_index: int | None = None
    chosen_timestamp: float | None = None

class NoteGenerator:
    """核心笔记生成器。

    流程：
    1) 分章：基于全部字幕让 LLM 输出章节边界
    2) 大章节再分章：达到字数阈值的章节会继续细分
    3) 分段：逐章节生成层次段落结构（仅构建文本与层级，不涉足截图）
    4) 截图：待所有段落就绪后统一生成九宫格缩略图
    5) 多模态选图：基于九宫格并发调用多模态 LLM 选择最佳帧
    6) 高清截图：按照选中的时间戳最终截取高清单帧
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
        self._thumb_sem = threading.Semaphore(max(1, self.cfg.screenshot.max_workers))
        self._shot_sem = threading.Semaphore(1)
        self._chapter_char_threshold = max(
            0,
            getattr(getattr(self.cfg, "note", None), "chapter_resegment_char_threshold", 0),
        )

    def _render_all_subtitles(self, meta: GenerationInputMeta) -> str:
        lines = []
        for seg in meta.subtitle.items:
            lines.append(f"{seg.line_no}\t[{seg.start_sec:.3f}-{seg.end_sec:.3f}]\t{seg.text}")
        return "\n".join(lines)

    @staticmethod
    def _render_segments_for_prompt(segs: List[SubtitleSegment]) -> str:
        lines = []
        for seg in segs:
            lines.append(f"{seg.line_no}\t[{seg.start_sec:.3f}-{seg.end_sec:.3f}]\t{seg.text}")
        return "\n".join(lines)

    @staticmethod
    def _summarize_paragraph_issue(issue: Dict[str, Any], err: Exception) -> str:
        parts = [f"校验异常：{err}"]
        missing = issue.get("missing") or []
        extra = issue.get("extra") or []
        dup = issue.get("duplicate") or []
        overlap = issue.get("overlap")
        boundary = issue.get("boundary_mismatch")
        if missing:
            parts.append(f"缺失行号样例：{missing[:10]}")
        if extra:
            parts.append(f"多余行号样例：{extra[:10]}")
        if dup:
            parts.append(f"重复行号样例：{dup[:10]}")
        if overlap:
            parts.append("段落时间存在重叠")
        if boundary:
            parts.append(f"边界不一致段落数量：{boundary}")
        return "\n".join(parts)

    @staticmethod
    def _paragraphs_to_prompt_json(model: ParagraphsSchema, seg_lookup: Dict[int, SubtitleSegment]) -> dict:
        def convert(node: ParagraphSchema) -> dict:
            data = node.model_dump()
            lines = data.get("lines") or []
            for line in lines:
                seg = seg_lookup.get(line.get("line_no"))
                if seg is not None:
                    line["text"] = seg.text
            children = node.children or []
            data["children"] = [convert(child) for child in children]
            if "optimized" not in data:
                data["optimized"] = []
            return data

        return {"paragraphs": [convert(p) for p in model.paragraphs]}

    def _prompt_for_chapters(self, meta: GenerationInputMeta) -> ChaptersSchema:
        """基于完整字幕生成“章节边界”。"""

        if meta.subtitle.items:
            _min_line_no = min(s.line_no for s in meta.subtitle.items)
            _max_line_no = max(s.line_no for s in meta.subtitle.items)
            _total_lines = len(meta.subtitle.items)
        else:
            _min_line_no = 0
            _max_line_no = 0
            _total_lines = 0

        if meta.subtitle.items:
            _min_sec = min(s.start_sec for s in meta.subtitle.items)
            _max_sec = max(s.end_sec for s in meta.subtitle.items)
            _total_sec = max(0.0, _max_sec - _min_sec)
        else:
            _min_sec = 0.0
            _max_sec = 0.0
            _total_sec = 0.0

        _int_total = int(_total_sec + 0.5)
        adv_min = max(3, min(20, -(-_int_total // 480)))
        adv_max_calc = max(3, min(20, _int_total // 180))
        adv_max = max(adv_min, adv_max_calc)
        min_required = 4 if (_total_sec >= 900 or _total_lines >= 120) else 1

        sys = SystemMessage(content=dedent(
            """
            你是一名专业的结构化编辑，任务是将“给定字幕的行号列表”划分为多个“章节”。

            严格要求（必须全部满足）：
            1) 行号来源：只使用“下面提供的字幕列表中真实出现过的行号”，禁止创造新行号、禁止估算/插值。
            2) 区间定义：章节行号区间为闭区间 [start_line_no, end_line_no]（两端均包含）。
            3) 相邻相接：相邻章节必须满足 start_line_no(i) = end_line_no(i-1) + 1。
               - 正确示例：[1-10], [11-20]
               - 错误示例：[1-10], [10-20]（共用 10 导致重叠）
            4) 全量覆盖：所有字幕行号必须被章节区间完整覆盖，无缺失、无重复、无共享行号。
            5) 无重叠且有序：任意两个章节不重叠，且按 start_line_no 升序排列。
            6) 首尾边界：第一个章节的 start_line_no 必须等于最小行号；最后一个章节的 end_line_no 必须等于最大行号。
            7) 标题生成规范：
               - 语义：标题必须基于该章节区间的实际内容进行“主题式总结”，类似书籍章节名，准确概括本章的核心主题/任务/结论。
               - 形式：使用简体中文名词短语或“主题+动作/结果”的短句；不得出现“第N章/Part/Chapter”等序号。
               - 取材：优先提取区间内的高频术语、实体名、动宾短语（如：部署脚本、索引优化、权限模型、异常处理、参数调优）。
               - 具体而不空泛：避免只给出“总结/结语/课程结尾/致谢/尾声”等泛化词，除非该区间内容本身就是对前文的系统性总结。
               - 禁止：纯英文/拼音/纯时间/纯行号/口语化寒暄（如“大家好”“谢谢观看”）/第一或第二人称称谓（如“我们/你”）/无意义形容（如“很棒的结束”）。
               - 示例：
                 • 内容：讲述异常类型与处理策略 → 标题：异常类型与处理策略
                 • 内容：展示从数据预处理到训练的关键步骤 → 标题：数据预处理与训练流程
            8) 输出：仅返回一个 JSON 对象，键为 chapters，元素为 {title, start_line_no, end_line_no}；不要输出任何解释或多余字段。

            计数校验（你在内部校验，不要输出过程）：
            - 令 len_i = end_line_no(i) - start_line_no(i) + 1。
            - 必须满足 Σ len_i = 字幕总行数（total_lines）。

            内部生成建议（你在内部执行，不要输出过程）：
            - 先拟定一个严格递增的 end_line_no 列表（不包含最大行号），再推导 start_line_no：
              start_1 = 最小行号；start_i = end_{i-1} + 1；最后一个 end_line_no = 最大行号。

            章节数量与时长规则（关键）：
            - 建议每章覆盖 3~8 分钟，优先按“主题边界”切分，而非机械均分。
            - 对于中长时长素材，应避免过少章节：若总时长 ≥ 15 分钟或总行数 ≥ 120，章节数不得少于最少下限（见用户侧给定范围）。
            - 禁止仅返回 1 个章节，除非素材极短（如总时长 < 6 分钟且总行数 < 40）。
            - 单章过长（>10 分钟）时应进一步细分，若无明显边界信号可依据主题/任务转换、术语变化、结构折入切分。

            内容边界信号（参考）：
            - 主题/任务/阶段切换；出现新的术语模块/子系统；从介绍 → 操作 → 结果/总结的阶段变化；
            - 提纲/转折提示词（首先/其次/然后/接着/因此/总结/回顾）；
            - 新示例/新问题/新小节开头；Q&A 段落；长时间停顿或场景显著变化（从字幕与时间上可感知）。
            """
        ))
        human = HumanMessage(content=dedent(
            f"""
            目标：将下面的完整字幕（含行号与时间戳）按“内容导向”拆分为文章章节（非等距、非机械平分）。
            请仅基于“已出现的行号”给出章节边界，并严格遵守系统约束与输出格式。

            合法行号范围：{_min_line_no} - {_max_line_no}
            字幕总行数：{_total_lines}
            视频总时长：约 {_total_sec:.1f} 秒（≈ {_total_sec / 60.0:.1f} 分）
            期望章节数范围：{adv_min} - {adv_max}（基于总时长估算）；硬性下限（若适用）：{min_required}
            注意：仅允许使用下方列表中真实出现的行号；不得使用未出现的行号；必须全量覆盖且无重叠，排序正确。如不确定边界，请依据“内容边界信号”适度合并或拆分，并满足“章节数量与时长规则”（禁止单章退回，除非素材极短）。

            {self._render_all_subtitles(meta)}
            """
        ))
        t_attempt = perf_counter()
        result = self.text_llm.structured_invoke(ChaptersSchema, [sys, human], json_mode=False)

        def _has_cjk(s: str) -> bool:
            return any('\u4e00' <= ch <= '\u9fff' for ch in s)

        for idx, ch in enumerate(result.chapters, start=1):
            if not ch.title or not _has_cjk(ch.title):
                ch.title = f"第{idx}章"

        try:
            self._validate_chapters(meta, result)
        except Exception as err:
            self.logger.error(
                "分章校验失败",
                extra={
                    "attempt": 1,
                    "chapters": len(result.chapters),
                    "cost_ms": int((perf_counter() - t_attempt) * 1000),
                    "error": str(err),
                },
            )
            raise

        self.logger.info(
            "分章完成",
            extra={
                "chapters": [chapter.title for chapter in result.chapters],
                "attempt": 1,
                "cost_ms": int((perf_counter() - t_attempt) * 1000),
            },
        )
        return result

    def _validate_chapters(self, meta: GenerationInputMeta, chs: ChaptersSchema) -> None:
        expected_lines = [s.line_no for s in meta.subtitle.items]
        if not expected_lines:
            raise ValueError("字幕为空，无法分章")
        expected_set = set(expected_lines)
        min_line = min(expected_lines)
        max_line = max(expected_lines)

        if not chs.chapters:
            raise ValueError("分章结果为空")

        used_lines: list[int] = []
        illegal_pairs: list[tuple[int, int, str]] = []
        non_exist: set[int] = set()
        for c in chs.chapters:
            if c.start_line_no > c.end_line_no:
                illegal_pairs.append((c.start_line_no, c.end_line_no, c.title))
                continue
            if c.start_line_no not in expected_set:
                non_exist.add(c.start_line_no)
            if c.end_line_no not in expected_set:
                non_exist.add(c.end_line_no)
            used_lines.extend(range(c.start_line_no, c.end_line_no + 1))

        if illegal_pairs:
            detail = ", ".join(f"{t}:[{s}-{e}]" for s, e, t in illegal_pairs[:5])
            raise ValueError(f"发现 {len(illegal_pairs)} 个非法区间（起始>结束）：{detail}")

        used_set = set(used_lines)
        extra_lines = sorted(list(used_set - expected_set))
        missing_lines = sorted(list(expected_set - used_set))
        cnt = Counter(used_lines)
        overlapped_lines = sorted([ln for ln, n in cnt.items() if n > 1])

        sorted_by_start = sorted(chs.chapters, key=lambda x: (x.start_line_no, x.end_line_no))
        not_sorted = list(chs.chapters) != list(sorted_by_start)

        errors: list[str] = []
        if overlapped_lines:
            sample = ", ".join(map(str, overlapped_lines[:10]))
            errors.append(f"章节行号存在重叠，样例：{sample}")
        if missing_lines:
            sample = ", ".join(map(str, missing_lines[:10]))
            errors.append(f"章节行号存在缺失，样例：{sample}")
        non_exist_all = sorted(list(non_exist | set(extra_lines)))
        if non_exist_all:
            sample = ", ".join(map(str, non_exist_all[:10]))
            errors.append(f"章节包含不存在的行号，样例：{sample}")
        if chs.chapters:
            first_start = min(c.start_line_no for c in chs.chapters)
            last_end = max(c.end_line_no for c in chs.chapters)
            if first_start != min_line or last_end != max_line:
                errors.append(
                    f"章节未覆盖完整字幕范围（应覆盖 {min_line}-{max_line}，实际 {first_start}-{last_end}）"
                )
        if not_sorted:
            errors.append("章节未按起始行号升序排列")

        if errors:
            raise ValueError("；".join(errors))

    def _select_lines(self, items: List[SubtitleSegment], start_line: int, end_line: int) -> List[SubtitleSegment]:
        return [s for s in items if start_line <= s.line_no <= end_line]

    def _segment_char_count(self, segs: List[SubtitleSegment]) -> int:
        return sum(len(s.text or "") for s in segs)

    def _should_resplit_chapter(self, segs: List[SubtitleSegment]) -> bool:
        if self._chapter_char_threshold <= 0:
            return False
        return self._segment_char_count(segs) > self._chapter_char_threshold

    def _build_sub_meta(
        self,
        meta: GenerationInputMeta,
        segs: List[SubtitleSegment],
    ) -> GenerationInputMeta:
        return GenerationInputMeta(
            video_path=meta.video_path,
            subtitle=SubtitleDocument(
                items=[s.model_copy(deep=True) for s in segs],
                source_path=meta.subtitle.source_path,
                format=meta.subtitle.format,
            ),
            params=dict(meta.params),
        )

    def _prompt_for_paragraphs(self, chapter_title: str, segs: List[SubtitleSegment]) -> ParagraphsSchema:
        if segs:
            _min_line = min(s.line_no for s in segs)
            _max_line = max(s.line_no for s in segs)
            _total_lines = len(segs)
            _min_sec = min(s.start_sec for s in segs)
            _max_sec = max(s.end_sec for s in segs)
        else:
            _min_line = 0
            _max_line = 0
            _total_lines = 0
            _min_sec = 0.0
            _max_sec = 0.0

        sys = SystemMessage(content=dedent(
            """
            你是一名专业的结构化编辑，任务是将章节内的字幕行划分为“段落”，并为每个段落生成 optimized。

            硬性约束（必须全部满足）：
            1) 行号一致性与连续性：仅使用输入列表中真实出现的行号；每个行号必须且仅出现一次；整体行号应构成连续整数序列（无间断、无缺失、无重复、无额外）。
            2) 顶层时间覆盖：顶层段落按时间升序且不重叠，联合完整覆盖章节时间范围 [min_sec, max_sec]（容差 ≤ 0.05s）。
               - 相邻段落应时间相接：start_sec(i) ≈ end_sec(i-1)；不得出现空洞或交叉。
            3) 边界=行集合：任意段落 start_sec = 行集合 start_sec 的最小值；end_sec = 行集合 end_sec 的最大值；段落内每行的 [start_sec, end_sec] 必须落在段落边界内（容差 ≤ 0.05s）。
            4) 排序与结构：段落按时间升序；lines 按行号升序。
            5) 输出契约：仅返回一个 JSON 对象（不出现额外文字/Markdown），唯一键为 paragraphs，值为 List[段落]。
               - 段落对象仅包含：title, start_sec, end_sec, lines, children, optimized。
               - 行对象仅包含：line_no, start_sec, end_sec, text。
               - lines.*.text 一律为空字符串 ""；禁止回传原文（程序会回填）。
               - children 若无子段落则为 []；optimized 为 List[str]。
            6) 字段最小化与结构：禁止输出未定义字段；lines/children 按要求排序。

            内省自检（输出前在你内部完成，不要打印过程）：
            - 行号校验：收集所有段落 lines 的 line_no 集合，应与输入全集完全相等；计数 Σ(唯一行) = total_lines；重复=0；缺失=0；额外=0。
            - 连续性校验：并集按升序，两两相邻差值恒为 1。
            - 时间校验：
              a) 顶层段落时间范围按升序且不重叠；
              b) 顶层覆盖：第一个段落 start_sec ≈ min_sec；最后一个段落 end_sec ≈ max_sec；
              c) 段落边界等于行集合的最小开始/最大结束；所有行时间落入段落边界。

            回退策略：若任何检验未通过，则仅返回 1 个顶层段落：
            - title:"整体内容概览"
            - start_sec=min_sec, end_sec=max_sec
            - lines：包含全部行（按行号升序），每行 start_sec/end_sec 与输入一致，text 一律为空串
            - children=[]；optimized 可生成为 1~N 条，但不得捏造事实

            optimized 说明：
            - 为每个段落生成 optimized（List[str]）：去除语气词、添加合理标点后，拼接成流畅句子；
            - 内容较长可自然分段；可使用 Markdown 标记重点（如 **加粗**、`行内代码`、列表等）；
            - 严禁捏造未出现在该段落字幕中的事实；保持术语与数字的准确性与时序逻辑。
            """
        ))
        content_lines = "\n".join(
            f"{s.line_no}\t[{s.start_sec:.3f}-{s.end_sec:.3f}]\t{s.text}" for s in segs
        )
        allowed_ids = "[" + ", ".join(str(s.line_no) for s in segs) + "]"
        human = HumanMessage(content=dedent(
            f"""
            章节标题：{chapter_title}
            本章节合法行号范围：{_min_line} - {_max_line}
            本章节字幕总行数：{_total_lines}
            本章节时间范围：[{_min_sec:.3f} - {_max_sec:.3f}]

            合法行号全集 S：{allowed_ids}
            注：S 为从 {_min_line} 到 {_max_line} 的连续整数集合。
            要求：仅可使用 S 中的行号；并且对所有段落 lines 的并集需“连续、无重复、无缺失、无额外”；段落之间时间不重叠且完整覆盖范围；段落边界应等于其 lines 的最小开始/最大结束；若无法满足全部硬性校验，请按回退策略输出单段结构。

            {content_lines}
            """
        ))
        with self._text_sem:
            return self.text_llm.structured_invoke(ParagraphsSchema, [sys, human], json_mode=False)

    def _extract_json_object(self, text: str) -> str:
        if text is None:
            raise ValueError("LLM 返回为空")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("LLM 返回为空白")

        fenced = re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", cleaned, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and start < end:
            return cleaned[start:end + 1].strip()
        return cleaned

    def _validate_time_coverage(self, segs: List[SubtitleSegment], paras: List[ParagraphSchema]) -> None:
        if not paras:
            raise ValueError("分段结果为空")
        eps = 0.05
        ranges = sorted([(p.start_sec, p.end_sec) for p in paras], key=lambda x: x[0])
        for i in range(1, len(ranges)):
            if ranges[i][0] < ranges[i - 1][1] - eps:
                raise ValueError("分段时间范围存在重叠")
        min_in = min(s.start_sec for s in segs)
        max_in = max(s.end_sec for s in segs)
        if abs(ranges[0][0] - min_in) > eps or abs(ranges[-1][1] - max_in) > eps:
            raise ValueError("分段时间范围未完整覆盖章节区间")

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

        for p in paras:
            for l in p.lines:
                if not (p.start_sec - eps <= l.start_sec <= l.end_sec <= p.end_sec + eps):
                    raise ValueError("段落内行时间越界")
            min_line_start = min(l.start_sec for l in p.lines)
            max_line_end = max(l.end_sec for l in p.lines)
            if abs(p.start_sec - min_line_start) > eps or abs(p.end_sec - max_line_end) > eps:
                raise ValueError("段落边界应等于该段落所有行的最小开始与最大结束")

    def _analyze_coverage_issue(self, segs: List[SubtitleSegment], paras: List[ParagraphSchema]) -> Dict[str, Any]:
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

    def _repair_paragraphs(
        self,
        chapter_title: str,
        segs: List[SubtitleSegment],
        previous: ParagraphsSchema,
        err: Exception,
        issue: Dict[str, Any],
    ) -> ParagraphsSchema:
        self.logger.info(
            "章节补偿请求",
            extra={
                "chapter_title": chapter_title,
                "segments": len(segs),
            },
        )
        summary = self._summarize_paragraph_issue(issue, err)
        missing_lines = issue.get("missing") or []
        extra_lines = issue.get("extra") or []
        duplicate_lines = issue.get("duplicate") or []
        seg_lookup = {s.line_no: s for s in segs}
        if segs:
            _min_line = min(seg_lookup)
            _max_line = max(seg_lookup)
            _total_lines = len(seg_lookup)
            _min_sec = min(s.start_sec for s in seg_lookup.values())
            _max_sec = max(s.end_sec for s in seg_lookup.values())
        else:
            _min_line = 0
            _max_line = 0
            _total_lines = 0
            _min_sec = 0.0
            _max_sec = 0.0
        allowed_ids = "[" + ", ".join(str(line) for line in sorted(seg_lookup)) + "]"
        content_lines = self._render_segments_for_prompt(segs)
        original_json = json.dumps(
            self._paragraphs_to_prompt_json(previous, seg_lookup),
            ensure_ascii=False,
            indent=2,
        )

        def _fmt(lines: List[int]) -> str:
            return ", ".join(map(str, lines[:30])) if lines else "无"

        sys = SystemMessage(content=dedent(
            """
            你是一名结构化编辑助手。请基于“上一次模型的 JSON 输出”做最小修复，并确保输出通过以下校验：
            1. 行号只能来自给定字幕且完全覆盖合法全集——不得缺失、重复或多余。
            2. 段落按时间升序覆盖整个章节时间范围；每个段落 start_sec/end_sec 必须分别等于该段落最小/最大行号的时间。
            3. 段落与行对象字段严格遵循约定结构，不得增删字段；lines 必须按行号升序。
            4. 输出只能是 JSON 对象 {"paragraphs": [...]}，严禁附加任何解释或 Markdown。
            5. optimized 字段是笔记核心摘要，请根据修复后的段落内容重新校正，可使用条列或短句说明重点，务必与段落语义一致。
            6. 允许合并/拆分段落，但请仅修改存在问题的部分；其它段落应尽量保持不变。
            7. 输出前请自检：
               - 汇总所有段落行号，与合法全集逐项对比确认完全一致。
               - 验证段落时间无重叠、覆盖完整区间。
               - 确认 optimized 与段落内容匹配且不可为空。
            """
        ))
        human = HumanMessage(content=dedent(
            f"""
            章节标题：{chapter_title}
            校验失败原因：{summary}

            需修复的行号问题：
            - 缺失行号样例：{_fmt(missing_lines)}
            - 多余行号样例：{_fmt(extra_lines)}
            - 重复行号样例：{_fmt(duplicate_lines)}

            上一次模型输出（已为每条 line_no 回填原字幕 text，供比对使用）：
            ```json
            {original_json}
            ```

            修复建议步骤：
            1) 调整问题段落的 lines，使行号集合与合法全集一致；如需合并/拆分段落请直接修改。
            2) 校正每个段落的 start_sec/end_sec，使其分别等于段落最小/最大行号的时间，并保证段落间无重叠、覆盖完整。
            3) 重写或更新受影响段落的 optimized 字段，使内容准确、结构清晰，不得留空。
            4) 保留其它已正确段落。

            供比对的字幕原文（行号/时间/文本）：
            合法行号范围：{_min_line} - {_max_line}
            字幕总行数：{_total_lines}
            时间范围：[{_min_sec:.3f} - {_max_sec:.3f}]
            合法行号全集 S：{allowed_ids}

            {content_lines}
            """
        ))
        return self.text_llm.structured_invoke(ParagraphsSchema, [sys, human], json_mode=False)

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
        with self._mm_sem:
            chosen = self.mm_llm.choose_index(instruction, str(grid_path))
        return int(chosen)

    def _convert_paragraph(
        self,
        ps: ParagraphSchema,
        line_lookup: Dict[int, SubtitleSegment] | None = None,
        backfill_text: bool = False,
    ) -> Paragraph:
        lines: List[SubtitleSegment] = []
        for l in ps.lines:
            txt = l.text or ""
            if backfill_text and (not txt):
                if line_lookup is not None:
                    seg = line_lookup.get(l.line_no)
                    if seg is not None:
                        txt = seg.text
            lines.append(
                SubtitleSegment(line_no=l.line_no, start_sec=l.start_sec, end_sec=l.end_sec, text=txt)
            )
        children = [self._convert_paragraph(c, line_lookup, backfill_text) for c in ps.children] if ps.children else []
        return Paragraph(
            title=ps.title,
            start_sec=ps.start_sec,
            end_sec=ps.end_sec,
            lines=lines,
            children=children,
            optimized=list(ps.optimized or []),
        )

    def _normalize_paragraphs(self, paras: List[ParagraphSchema]) -> None:
        def _norm_list(nodes: List[ParagraphSchema]) -> None:
            for p in nodes:
                if p.lines:
                    p.lines.sort(key=lambda l: l.line_no)
                    min_line_start = min(l.start_sec for l in p.lines)
                    max_line_end = max(l.end_sec for l in p.lines)
                    p.start_sec = min_line_start
                    p.end_sec = max_line_end
                if p.children:
                    _norm_list(p.children)
            nodes.sort(key=lambda x: (x.start_sec, x.end_sec))

        _norm_list(paras)

    def _schedule_paragraph_render_tasks(
        self,
        paragraph: Paragraph,
        base_dir: Path,
        label: str | None,
        chapter_index: int,
        chapter_path: str,
        chapter_title: str,
        para_index: int,
    ) -> List[ParagraphRenderTask]:
        base_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            ParagraphRenderTask(
                chapter_index=chapter_index,
                chapter_path=chapter_path,
                chapter_title=chapter_title,
                paragraph_index=para_index,
                paragraph=paragraph,
                base_dir=base_dir,
                label=label,
            )
        ]
        if paragraph.children:
            for si, child in enumerate(paragraph.children, start=1):
                child_dir = base_dir / f"sub_{si}"
                sub_label = f"{label}/sub_{si}" if label else f"sub_{si}"
                tasks.extend(
                    self._schedule_paragraph_render_tasks(
                        child,
                        child_dir,
                        sub_label,
                        chapter_index,
                        chapter_path,
                        chapter_title,
                        para_index,
                    )
                )
        return tasks

    def _process_leaf_chapter(
        self,
        index_path: Tuple[int, ...],
        title: str,
        segs: List[SubtitleSegment],
        meta: GenerationInputMeta,
        task_out_dir: Path,
    ) -> tuple[Chapter, List[ParagraphRenderTask]]:
        chapter_index = index_path[0]
        chapter_path = ".".join(str(i) for i in index_path)
        chapter_dir = task_out_dir / ("chapter_" + "_".join(str(i) for i in index_path))
        try:
            chapter_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        pgs = self._prompt_for_paragraphs(title, segs)
        validated_from = "initial"
        try:
            self._normalize_paragraphs(pgs.paragraphs)
            self._validate_time_coverage(segs, pgs.paragraphs)
        except Exception as err:
            issue = self._analyze_coverage_issue(segs, pgs.paragraphs)
            self.logger.info(
                "章节校验失败，准备补偿",
                extra={
                    "chapter_index": chapter_index,
                    "chapter_path": chapter_path,
                    "paragraphs": len(pgs.paragraphs),
                    "error": str(err),
                    **issue,
                },
            )
            attempts = 0
            repaired_success = False
            current = pgs
            last_error: Exception | None = None
            while attempts < 3 and not repaired_success:
                attempts += 1
                repaired = self._repair_paragraphs(title, segs, current, err if attempts == 1 else last_error or err, issue)
                current = repaired
                self._normalize_paragraphs(current.paragraphs)
                try:
                    self._validate_time_coverage(segs, current.paragraphs)
                    repaired_success = True
                    validated_from = "repair" if attempts == 1 else f"repair_retry_{attempts}"
                except Exception as retry_err:
                    last_error = retry_err
                    issue = self._analyze_coverage_issue(segs, current.paragraphs)
                    if attempts >= 3:
                        self.logger.error(
                            "章节补偿连续失败",
                            extra={
                                "chapter_index": chapter_index,
                                "chapter_path": chapter_path,
                                "attempt": attempts,
                                "error": str(retry_err),
                                **issue,
                            },
                        )
                        raise ValueError("章节补偿校验失败") from retry_err
                    self.logger.info(
                        "章节补偿再次失败，准备继续修复",
                        extra={
                            "chapter_index": chapter_index,
                            "chapter_path": chapter_path,
                            "attempt": attempts,
                            "error": str(retry_err),
                            **issue,
                        },
                    )
            pgs = current

        self.logger.info(
            "章节校验通过",
            extra={
                "chapter_index": chapter_index,
                "chapter_path": chapter_path,
                "paragraphs": len(pgs.paragraphs),
                "source": validated_from,
            },
        )

        mode = getattr(getattr(self.cfg, "note", None), "mode", "subtitle")
        backfill = mode == "subtitle"
        line_lookup = {s.line_no: s for s in segs}

        paragraphs: List[Paragraph] = []
        tasks: List[ParagraphRenderTask] = []
        for idx, ps in enumerate(pgs.paragraphs, start=1):
            para = self._convert_paragraph(ps, line_lookup=line_lookup, backfill_text=backfill)
            base_dir = chapter_dir / f"para_{idx}"
            paragraphs.append(para)
            tasks.extend(
                self._schedule_paragraph_render_tasks(
                    para,
                    base_dir,
                    None,
                    chapter_index,
                    chapter_path,
                    title,
                    idx,
                )
            )

        chapter = Chapter(
            title=title,
            start_sec=min(s.start_sec for s in segs),
            end_sec=max(s.end_sec for s in segs),
            paragraphs=paragraphs,
            children=[],
        )
        return chapter, tasks

    def _process_chapter_node(
        self,
        index_path: Tuple[int, ...],
        title: str,
        segs: List[SubtitleSegment],
        meta: GenerationInputMeta,
        task_out_dir: Path,
    ) -> tuple[Chapter, List[ParagraphRenderTask]]:
        if not segs:
            raise ValueError(f"章节无内容：{title}")

        chapter_index = index_path[0]
        chapter_path = ".".join(str(i) for i in index_path)
        char_count = self._segment_char_count(segs)
        start_sec = min(s.start_sec for s in segs)
        end_sec = max(s.end_sec for s in segs)

        self.logger.info(
            "处理章节",
            extra={
                "chapter_index": chapter_index,
                "chapter_path": chapter_path,
                "title": title,
                "segments": len(segs),
                "char_count": char_count,
            },
        )

        if self._should_resplit_chapter(segs):
            self.logger.info(
                "章节字符数超出阈值，启动细分",
                extra={
                    "chapter_index": chapter_index,
                    "chapter_path": chapter_path,
                    "char_count": char_count,
                    "threshold": self._chapter_char_threshold,
                },
            )
            sub_meta = self._build_sub_meta(meta, segs)
            sub_boundaries = self._prompt_for_chapters(sub_meta)
            parent_range = (min(s.line_no for s in segs), max(s.line_no for s in segs))
            ranges = [(c.start_line_no, c.end_line_no) for c in sub_boundaries.chapters]
            has_effective_split = len(ranges) > 1 and any(r != parent_range for r in ranges)

            if has_effective_split:
                children: List[Chapter] = []
                tasks: List[ParagraphRenderTask] = []
                for child_idx, child_cb in enumerate(sub_boundaries.chapters, start=1):
                    child_segs = self._select_lines(segs, child_cb.start_line_no, child_cb.end_line_no)
                    if not child_segs:
                        raise ValueError(f"子章节无内容：{child_cb}")
                    child, child_tasks = self._process_chapter_node(
                        index_path + (child_idx,),
                        child_cb.title,
                        child_segs,
                        meta,
                        task_out_dir,
                    )
                    children.append(child)
                    tasks.extend(child_tasks)

                chapter = Chapter(
                    title=title,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    paragraphs=[],
                    children=children,
                )
                return chapter, tasks

            self.logger.info(
                "子章节拆分未产生更小区间，继续按段落生成",
                extra={
                    "chapter_index": chapter_index,
                    "chapter_path": chapter_path,
                    "char_count": char_count,
                    "threshold": self._chapter_char_threshold,
                },
            )

        return self._process_leaf_chapter(index_path, title, segs, meta, task_out_dir)

    def _process_chapter(
        self,
        ci: int,
        cb: ChapterBoundary,
        meta: GenerationInputMeta,
        task_out_dir: Path,
    ) -> tuple[Chapter, List[ParagraphRenderTask]]:
        segs = self._select_lines(meta.subtitle.items, cb.start_line_no, cb.end_line_no)
        return self._process_chapter_node((ci,), cb.title, segs, meta, task_out_dir)

    def _stage_thumbnail_grids(
        self,
        tasks: List[ParagraphRenderTask],
        meta: GenerationInputMeta,
    ) -> None:
        video_path = Path(meta.video_path)
        for task in tasks:
            para = task.paragraph
            base_dir = task.base_dir
            base_dir.mkdir(parents=True, exist_ok=True)
            timestamps = generate_grid_timestamps(para.start_sec, para.end_sec, self.cfg.screenshot)
            task.timestamps = list(timestamps)
            grid_path = base_dir / "grid.jpg"
            thumbs_dir = base_dir / "thumbs"
            t_s = perf_counter()
            try:
                with self._thumb_sem:
                    self.screenshotter.compose_grid_one_shot(
                        video_path,
                        timestamps,
                        grid_path,
                        cols=self.cfg.screenshot.grid_columns,
                        rows=self.cfg.screenshot.grid_rows,
                        width=self.cfg.screenshot.low_width,
                        height=self.cfg.screenshot.low_height,
                    )
            except Exception as e:
                thumbs_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(
                    "一把流九宫格失败，将回退逐帧",
                    extra={
                        "chapter_index": task.chapter_index,
                        "chapter_path": task.chapter_path,
                        "para_index": task.paragraph_index,
                        "sub_index": task.label or "",
                        "error": str(e),
                    },
                )
                with self._thumb_sem:
                    thumbs = self.screenshotter.capture_thumbs(video_path, timestamps, thumbs_dir)
                    self.screenshotter.compose_grid(thumbs, grid_path)
            cost_ms = int((perf_counter() - t_s) * 1000)
            self.logger.info(
                "生成九宫格完成",
                extra={
                    "chapter_index": task.chapter_index,
                    "chapter_path": task.chapter_path,
                    "para_index": task.paragraph_index,
                    "sub_index": task.label or "",
                    "grid": str(grid_path),
                    "cost_ms": cost_ms,
                },
            )
            task.grid_path = grid_path
            image = self._ensure_paragraph_image(task.paragraph)
            image.grid_image_path = grid_path
            image.grid_timestamps_sec = list(timestamps)

    def _stage_multimodal_selection(self, tasks: List[ParagraphRenderTask]) -> None:
        if not tasks:
            return
        candidates = [t for t in tasks if t.grid_path and t.timestamps]
        total = len(candidates)
        if total == 0:
            return
        self.logger.info(
            "多模态选图阶段开始",
            extra={"candidates": total},
        )
        futures: Dict[Any, ParagraphRenderTask] = {}
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.mm_llm.concurrency)) as ex:
            for task in candidates:
                fut = ex.submit(
                    self._choose_best_frame,
                    task.paragraph,
                    task.grid_path,
                    task.chapter_index,
                    task.paragraph_index,
                    task.label,
                )
                futures[fut] = task
                self.logger.info(
                    "多模态选图排队",
                    extra={
                        "chapter_index": task.chapter_index,
                        "chapter_path": task.chapter_path,
                        "para_index": task.paragraph_index,
                        "sub_index": task.label or "",
                        "grid": str(task.grid_path),
                    },
                )
            completed = 0
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    chosen = fut.result()
                except Exception as exc:
                    self.logger.error(
                        "多模态选图失败，回退默认索引5",
                        extra={
                            "chapter_index": task.chapter_index,
                            "chapter_path": task.chapter_path,
                            "para_index": task.paragraph_index,
                            "sub_index": task.label or "",
                            "error": str(exc),
                        },
                    )
                    chosen = 5
                    task.timestamps = task.timestamps or []
                    if not task.timestamps:
                        task.timestamps = [0.0] * 9
                chosen = max(1, min(9, int(chosen)))
                if chosen > len(task.timestamps):
                    task.timestamps.extend([task.timestamps[-1] if task.timestamps else 0.0] * (chosen - len(task.timestamps)))
                task.chosen_index = chosen
                task.chosen_timestamp = task.timestamps[chosen - 1]
                image = self._ensure_paragraph_image(task.paragraph)
                image.chosen_index = chosen
                image.chosen_timestamp_sec = task.chosen_timestamp
                completed += 1
                self.logger.info(
                    "多模态选图完成",
                    extra={
                        "chapter_index": task.chapter_index,
                        "chapter_path": task.chapter_path,
                        "para_index": task.paragraph_index,
                        "sub_index": task.label or "",
                        "chosen_index": chosen,
                        "timestamp_sec": task.chosen_timestamp,
                        "completed": completed,
                        "total": total,
                    },
                )
        self.logger.info(
            "多模态选图阶段完成",
            extra={"completed": total, "total": total},
        )

    def _stage_high_quality_capture(
        self,
        tasks: List[ParagraphRenderTask],
        meta: GenerationInputMeta,
    ) -> None:
        video_path = Path(meta.video_path)
        note_cfg = getattr(self.cfg, "note", None)
        hi_root = getattr(note_cfg, "screenshot_dir", None)
        for task in tasks:
            if task.chosen_index is None or task.chosen_timestamp is None:
                continue
            timestamps = task.timestamps or []
            if not timestamps:
                continue
            raw_name = f"{Path(meta.video_path).stem}_{task.paragraph.title}_{self._format_ts_hhmmss(task.chosen_timestamp)}"
            hi_name = self._sanitize_filename(raw_name) + ".jpg"
            out_root = Path(hi_root) if hi_root else task.base_dir
            try:
                out_root.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            hi_path = out_root / hi_name
            t_hq = perf_counter()
            with self._shot_sem:
                self.screenshotter.capture_high_quality(video_path, task.chosen_timestamp, hi_path)
            cost_ms = int((perf_counter() - t_hq) * 1000)
            self.logger.info(
                "高清重拍完成",
                extra={
                    "chapter_index": task.chapter_index,
                    "chapter_path": task.chapter_path,
                    "para_index": task.paragraph_index,
                    "sub_index": task.label or "",
                    "chosen_index": task.chosen_index,
                    "timestamp_sec": task.chosen_timestamp,
                    "hi_image": str(hi_path),
                    "cost_ms": cost_ms,
                },
            )
            image = self._ensure_paragraph_image(task.paragraph)
            image.hi_res_image_path = hi_path

    def _ensure_paragraph_image(self, para: Paragraph) -> ParagraphImage:
        if para.image is None:
            para.image = ParagraphImage()
        return para.image

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        sanitized = re.sub(r"[\\\\/\\\\?%\\*:|\\\"<>]", "", name)
        sanitized = re.sub(r"\\s+", "_", sanitized).strip("._ ")
        return sanitized[:150] or "untitled"

    @staticmethod
    def _format_ts_hhmmss(ts: float) -> str:
        t = max(0, int(ts))
        hh = t // 3600
        mm = (t % 3600) // 60
        ss = t % 60
        return f"{hh:02d}{mm:02d}{ss:02d}"

    def generate(self, meta: GenerationInputMeta, task_out_dir: Path) -> Note:
        chs = self._prompt_for_chapters(meta)
        chapters: List[Chapter] = [None] * len(chs.chapters)
        tasks: List[ParagraphRenderTask] = []

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.text_llm.concurrency)) as ex:
            future_map: Dict[Any, int] = {}
            for ci, cb in enumerate(chs.chapters, start=1):
                fut = ex.submit(self._process_chapter, ci, cb, meta, task_out_dir)
                future_map[fut] = ci
            try:
                for fut in as_completed(future_map):
                    ci = future_map[fut]
                    chapter_meta = chs.chapters[ci - 1]
                    try:
                        chapter, chapter_tasks = fut.result()
                    except Exception as exc:
                        self.logger.error(
                            "章节生成失败，终止任务",
                            extra={
                                "chapter_index": ci,
                                "chapter_title": getattr(chapter_meta, "title", ""),
                                "error": str(exc),
                            },
                        )
                        for pending, _ci in future_map.items():
                            if pending is not fut:
                                pending.cancel()
                        raise RuntimeError(
                            f"章节 {ci} 生成失败：{getattr(chapter_meta, 'title', '')}") from exc
                    chapters[ci - 1] = chapter
                    tasks.extend(chapter_tasks)
            except Exception:
                raise

        note = Note(
            video_path=Path(meta.video_path),
            subtitle=meta.subtitle,
            chapters=chapters,
            meta={
                "text_llm_model": self.cfg.text_llm.model,
                "mm_llm_model": self.cfg.mm_llm.model,
            },
        )

        if tasks:
            ordered_tasks = sorted(
                tasks,
                key=lambda t: (
                    tuple(int(p) for p in t.chapter_path.split(".")),
                    t.paragraph_index,
                    t.label or "",
                ),
            )
        else:
            ordered_tasks = []

        self.logger.info(
            "章节结构生成完成",
            extra={"chapters": len(note.chapters), "paragraph_tasks": len(ordered_tasks)},
        )

        if ordered_tasks:
            self._stage_thumbnail_grids(ordered_tasks, meta)
            self._stage_multimodal_selection(ordered_tasks)
            self._stage_high_quality_capture(ordered_tasks, meta)

        self.logger.info("组装 Note 完成", extra={"chapters": len(note.chapters)})
        return note
