from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from textwrap import dedent
from time import perf_counter, sleep
from typing import Any, Dict, List, Tuple

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
    ParagraphBoundary,
    ParagraphImage,
    ParagraphLine,
    ParagraphOptimizationSchema,
    ParagraphPlan,
    ParagraphSchema,
    ParagraphsSchema,
)
from core.screenshot.ffmpeg import Screenshotter
from core.screenshot.grid import generate_grid_timestamps
from core.subtitles.models import SubtitleDocument, SubtitleSegment

from core.utils.retry import RetryPolicy

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
        self._chapter_retry = RetryPolicy()
        self._paragraph_retry = RetryPolicy()
        # 截图相关信号量
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
            - 在确认每个 end_line_no 前逐行核对该行字幕与下一行是否仍属于同一句话（如末尾为逗号、顿号、冒号、引号未闭合或语意未收束）；若仍在同一句中，必须把整句保留在同一章节，再微调边界。

            章节数量与时长规则（关键）：
            - 建议每章覆盖 3~8 分钟，优先按“主题边界”切分，而非机械均分。
            - 对于中长时长素材，应避免过少章节：若总时长 ≥ 15 分钟或总行数 ≥ 120，章节数不得少于最少下限（见用户侧给定范围）。
            - 禁止仅返回 1 个章节，除非素材极短（如总时长 < 6 分钟且总行数 < 40）。
            - 单章过长（>10 分钟）时应进一步细分，若无明显边界信号可依据主题/任务转换、术语变化、结构折入切分。

            内容边界信号（参考）：
            - 完整句末标点或语义终止信号（。！？……或明显总结、停顿）；若仅出现逗号、顿号、冒号、悬而未决的引号等中途停顿，视为同一句，禁止在此断开。
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
        max_attempts = self._chapter_retry.max_retries + 1
        attempts = 0
        last_err: Exception | None = None

        def _has_cjk(s: str) -> bool:
            return any('\u4e00' <= ch <= '\u9fff' for ch in s)

        while attempts < max_attempts:
            attempts += 1
            t_attempt = perf_counter()
            result: ChaptersSchema | None = None
            try:
                result = self.text_llm.structured_invoke(ChaptersSchema, [sys, human], json_mode=False)

                for idx, ch in enumerate(result.chapters, start=1):
                    if not ch.title or not _has_cjk(ch.title):
                        ch.title = f"第{idx}章"

                self._validate_chapters(meta, result)

                cost_ms = int((perf_counter() - t_attempt) * 1000)
                self.logger.info(
                    "分章完成",
                    extra={
                        "chapters": [chapter.title for chapter in result.chapters],
                        "attempt": attempts,
                        "cost_ms": cost_ms,
                    },
                )
                return result
            except Exception as err:
                last_err = err
                cost_ms = int((perf_counter() - t_attempt) * 1000)
                log_extra = {
                    "attempt": attempts,
                    "max": max_attempts,
                    "cost_ms": cost_ms,
                    "error": str(err),
                }
                if result is not None:
                    log_extra["chapters"] = len(result.chapters)

                if attempts >= max_attempts:
                    self.logger.error("分章校验失败", extra=log_extra)
                    raise

                self.logger.warning("分章失败，准备重试", extra=log_extra)
                sleep(2)

        assert last_err is not None
        raise last_err

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
        if not segs:
            raise ValueError("章节无字幕，无法分段")

        seg_lookup = {s.line_no: s for s in segs}
        try:
            plan = self._prompt_paragraph_plan(chapter_title, segs)
            paragraphs = self._build_paragraph_schemas(plan.paragraphs, seg_lookup)
            return ParagraphsSchema(paragraphs=paragraphs)
        except Exception as err:
            self.logger.error(
                "分段计划生成失败，回退单段结构",
                extra={
                    "chapter_title": chapter_title,
                    "segments": len(segs),
                    "error": str(err),
                },
            )
            return self._fallback_paragraphs(chapter_title, segs)

    def _prompt_paragraph_plan(self, chapter_title: str, segs: List[SubtitleSegment]) -> ParagraphPlan:
        min_line = min(s.line_no for s in segs)
        max_line = max(s.line_no for s in segs)
        total_lines = len(segs)
        content_lines = self._render_segments_for_prompt(segs)
        allowed_ids = "[" + ", ".join(str(s.line_no) for s in segs) + "]"

        sys = SystemMessage(content=dedent(
            """
            你是一名结构化编辑，任务是将给定字幕划分为多个段落，仅输出段落的标题与起止行号。

            必须遵守：
            1) 只能使用提供的行号，start_line_no ≤ end_line_no；数字必须为整数。
            2) 段落按 start_line_no 升序排列，相邻段落需满足 start_line_no(i) = end_line_no(i-1) + 1。
            3) 所有字幕行需被完整覆盖且仅出现一次；首段起点等于最小行号，末段终点等于最大行号。
            4) 段落标题需基于该行号区间的内容给出简体中文短句，突出该段主题，禁止出现“第N段”或纯序号。
            5) 输出严格为 JSON 对象 {"paragraphs": [{"title": str, "start_line_no": int, "end_line_no": int}, ...]}，不得添加其他字段或文字。
            6) 至少返回 1 个段落，若确实无法判断边界，请将全部行归入单段并给出能够涵盖全段的标题。
            7) 禁止把同一句话拆分到不同段落；若字幕跨多行组成同一句或同一说明（含颜色、方向、图形等细节），必须保持在同一段落内，并调整边界以容纳整句。

            质量指南：
            - 请根据内容实际划分合理段落数量，可先估算每段覆盖约 10-20 行字幕，再结合语义微调。
            - 建立在完整句子基础上划分边界，若边界行与相邻行仍共同构成一句话或引用尚未闭合，需调整使整句留在同一段。
            - 留意语义转折信号（如“首先/接着/最后”“接下来”“总结一下”等），以及主题从介绍 → 举例 → 总结的变化。
            - 段落需在完整观点处结束，避免把未展开的句子单独拆成新段。
            - 可参考画面切换或明显停顿，但优先以内容主题为依据，确保上下段衔接自然。
            - 标题要准确概括该段核心信息，使用简体中文短句，突出主题与作用。
            """
        ))
        human = HumanMessage(content=dedent(
            f"""
            章节标题：{chapter_title}
            合法行号范围：{min_line} - {max_line}
            字幕总行数：{total_lines}
            合法行号全集 S：{allowed_ids}

            请在上述范围内规划段落列表，仅包含标题与行号区间：

            {content_lines}
            """
        ))

        max_attempts = self._paragraph_retry.max_retries + 1
        attempts = 0
        last_err: Exception | None = None

        while attempts < max_attempts:
            attempts += 1
            t_attempt = perf_counter()
            plan: ParagraphPlan | None = None
            try:
                plan = self.text_llm.structured_invoke(ParagraphPlan, [sys, human], json_mode=False)
                self._validate_paragraph_plan(segs, plan)
                cost_ms = int((perf_counter() - t_attempt) * 1000)
                self.logger.info(
                    "分段计划生成完成",
                    extra={
                        "chapter_title": chapter_title,
                        "paragraphs": len(plan.paragraphs),
                        "attempt": attempts,
                        "cost_ms": cost_ms,
                    },
                )
                return plan
            except Exception as err:
                last_err = err
                cost_ms = int((perf_counter() - t_attempt) * 1000)
                log_extra = {
                    "chapter_title": chapter_title,
                    "segments": len(segs),
                    "attempt": attempts,
                    "max": max_attempts,
                    "cost_ms": cost_ms,
                    "error": str(err),
                }
                if plan is not None:
                    log_extra["paragraphs"] = len(plan.paragraphs)

                if attempts >= max_attempts:
                    self.logger.error("分段计划校验失败", extra=log_extra)
                    raise

                self.logger.warning("分段计划生成失败，准备重试", extra=log_extra)
                sleep(2)

        assert last_err is not None
        raise last_err

    def _validate_paragraph_plan(self, segs: List[SubtitleSegment], plan: ParagraphPlan) -> None:
        if not plan.paragraphs:
            raise ValueError("分段计划为空")

        expected_lines = [s.line_no for s in segs]
        min_line = min(expected_lines)
        max_line = max(expected_lines)
        expected_set = set(expected_lines)

        used_lines: list[int] = []
        prev_end: int | None = None
        for node in plan.paragraphs:
            if node.start_line_no > node.end_line_no:
                raise ValueError("存在 start_line_no > end_line_no 的段落")
            if node.start_line_no not in expected_set or node.end_line_no not in expected_set:
                raise ValueError("段落行号不在合法范围内")
            if prev_end is not None and node.start_line_no != prev_end + 1:
                raise ValueError("段落之间未连续衔接")
            used_lines.extend(range(node.start_line_no, node.end_line_no + 1))
            prev_end = node.end_line_no

        if plan.paragraphs[0].start_line_no != min_line or plan.paragraphs[-1].end_line_no != max_line:
            raise ValueError("段落计划未覆盖完整范围")

        used_counter = Counter(used_lines)
        if sorted(used_counter.keys()) != sorted(expected_set):
            raise ValueError("段落计划与原始行号集合不一致")
        duplicates = [ln for ln, cnt in used_counter.items() if cnt > 1]
        if duplicates:
            raise ValueError(f"存在重复覆盖的行号：{duplicates[:10]}")

    def _build_paragraph_schemas(
        self,
        boundaries: List[ParagraphBoundary],
        seg_lookup: Dict[int, SubtitleSegment],
    ) -> List[ParagraphSchema]:
        return [self._build_paragraph_schema(boundary, seg_lookup) for boundary in boundaries]

    def _build_paragraph_schema(
        self,
        boundary: ParagraphBoundary,
        seg_lookup: Dict[int, SubtitleSegment],
    ) -> ParagraphSchema:
        segs: List[SubtitleSegment] = []
        for line_no in range(boundary.start_line_no, boundary.end_line_no + 1):
            seg = seg_lookup.get(line_no)
            if seg is None:
                raise ValueError(f"找不到行号 {line_no} 对应的字幕")
            segs.append(seg)
        if not segs:
            raise ValueError("段落无字幕内容")

        lines = [
            ParagraphLine(
                line_no=seg.line_no,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                text="",
            )
            for seg in segs
        ]

        return ParagraphSchema(
            title=boundary.title,
            start_sec=segs[0].start_sec,
            end_sec=segs[-1].end_sec,
            lines=lines,
            children=[],
        )

    def _fallback_paragraphs(self, chapter_title: str, segs: List[SubtitleSegment]) -> ParagraphsSchema:
        seg_lookup = {s.line_no: s for s in segs}
        boundary = ParagraphBoundary(
            title="整体内容概览",
            start_line_no=min(seg_lookup.keys()),
            end_line_no=max(seg_lookup.keys()),
        )
        paragraph = self._build_paragraph_schema(boundary, seg_lookup)
        self.logger.warning(
            "分段回退为单段",
            extra={
                "chapter_title": chapter_title,
                "segments": len(segs),
            },
        )
        return ParagraphsSchema(paragraphs=[paragraph])

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

    def _choose_best_frame(self, para: Paragraph, grid_path: Path) -> int:
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

    def _generate_ai_notes_for_paragraphs(
        self,
        chapter_title: str,
        paragraphs: List[ParagraphSchema],
        line_lookup: Dict[int, SubtitleSegment],
    ) -> None:
        for ps in paragraphs:
            ps.optimized = self._optimize_paragraph_notes(chapter_title, ps, line_lookup)
            if ps.children:
                self._generate_ai_notes_for_paragraphs(chapter_title, ps.children, line_lookup)

    def _optimize_paragraph_notes(
        self,
        chapter_title: str,
        paragraph: ParagraphSchema,
        line_lookup: Dict[int, SubtitleSegment],
    ) -> List[str]:
        lines_content: List[str] = []
        for line in paragraph.lines:
            seg = line_lookup.get(line.line_no)
            if seg is None:
                continue
            text = (seg.text or "").strip()
            if not text:
                continue
            lines_content.append(text)

        if not lines_content:
            return []

        lines_text = "\n".join(lines_content)

        sys = SystemMessage(content=dedent(
            """
            你是一名字幕整理助手，任务是将给定字幕按语义整理为连贯的笔记语句。

            严格要求：
            1) 仅使用原字幕中的内容，禁止扩写新信息，也不要加入“本段概述”这类与字幕无关的说明。
            2) 可以合并多行字幕为完整句子、补全标点或调整语序，但必须保持原意准确。
            3) 输出为 JSON 对象 {"optimized": [...]}，列表元素按语义顺序排列，每个元素必须是实际的正文句子或要点；禁止输出“无内容/本段概述”等占位语。
            4) 若字幕行内容极其空泛（如只有寒暄），也要保留原文，不得擅自总结或省略。
            5) 允许使用轻量 Markdown（如列表、加粗）强调重点，但总体应简洁，避免冗余解释。
            """
        ))
        human = HumanMessage(content=dedent(
            f"""
            请根据以下字幕正文整理为书面化的笔记句子，合并重复或碎片化的表达，但确保涵盖全部信息：
            {lines_text}
            """
        ))

        max_attempts = self._paragraph_retry.max_retries + 1
        attempts = 0
        last_err: Exception | None = None

        while attempts < max_attempts:
            attempts += 1
            try:
                result = self.text_llm.structured_invoke(
                    ParagraphOptimizationSchema,
                    [sys, human],
                    json_mode=False,
                )
                optimized = [item.strip() for item in result.optimized if item.strip()]
                self.logger.info(
                    "段落优化完成",
                    extra={
                        "chapter_title": chapter_title,
                        "paragraph_title": paragraph.title,
                        "lines": len(lines_content),
                        "items": len(optimized),
                        "attempt": attempts,
                    },
                )
                return optimized
            except Exception as err:
                last_err = err
                log_extra = {
                    "chapter_title": chapter_title,
                    "paragraph_title": paragraph.title,
                    "lines": len(lines_content),
                    "attempt": attempts,
                    "max": max_attempts,
                    "error": str(err),
                }
                if attempts >= max_attempts:
                    self.logger.error("段落优化连续失败，将返回空结果", extra=log_extra)
                    break
                self.logger.warning("段落优化失败，准备重试", extra=log_extra)
                sleep(2)

        if last_err is not None:
            self.logger.debug("段落优化最终异常", extra={"error": str(last_err)})
        return []

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
        self._normalize_paragraphs(pgs.paragraphs)
        self._validate_time_coverage(segs, pgs.paragraphs)
        self.logger.info(
            "章节校验通过",
            extra={
                "chapter_index": chapter_index,
                "chapter_path": chapter_path,
                "paragraphs": len(pgs.paragraphs),
            },
        )

        mode = getattr(getattr(self.cfg, "note", None), "mode", "subtitle")
        backfill = mode == "subtitle"
        line_lookup = {s.line_no: s for s in segs}

        if mode == "optimized":
            self._generate_ai_notes_for_paragraphs(title, pgs.paragraphs, line_lookup)

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

            raw_timestamps = list(
                generate_grid_timestamps(para.start_sec, para.end_sec, self.cfg.screenshot)
            )
            tol = 0.05
            align_result = self.screenshotter.align_timestamps_to_frames(video_path, raw_timestamps)
            if align_result:
                aligned_ts, tol_hint = align_result
                if len(aligned_ts) == len(raw_timestamps):
                    raw_timestamps = aligned_ts
                    tol = max(0.003, min(0.05, tol_hint))
            timestamps = list(raw_timestamps)
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
                        tol=tol,
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
        video_stem = self._sanitize_filename(video_path.stem, allow_empty=True)
        note_cfg = getattr(self.cfg, "note", None)
        hi_root = getattr(note_cfg, "screenshot_dir", None)
        for task in tasks:
            if task.chosen_index is None or task.chosen_timestamp is None:
                continue
            timestamps = task.timestamps or []
            if not timestamps:
                continue
            name_parts = [
                video_stem,
                self._sanitize_filename(task.paragraph.title or "", allow_empty=True),
                self._format_ts_hhmmss(task.chosen_timestamp),
            ]
            hi_base = "_".join(part for part in name_parts if part)
            # 使用 PNG 保持无损画质，避免重压缩造成信息丢失
            hi_name = self._sanitize_filename(hi_base) + ".png"
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
    def _sanitize_filename(name: str, *, allow_empty: bool = False) -> str:
        """过滤非法字符并统一空格为下划线，保持可追溯的命名。"""
        # 先将常见特殊符号（例如 #、*）统一替换为下划线，避免在不同平台触发解析问题
        sanitized = re.sub(r"[#*]+", "_", name)
        # 再移除文件系统不允许的符号
        sanitized = re.sub(r"[\\/?%:|\"<>]", "", sanitized)
        sanitized = re.sub(r"[\s\u3000]+", "_", sanitized)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = sanitized.strip("._ ")
        if not sanitized:
            return "" if allow_empty else "untitled"
        return sanitized[:150]

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
