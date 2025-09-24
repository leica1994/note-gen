from __future__ import annotations

import logging
import threading
from collections import Counter
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
        """基于完整字幕生成“章节边界”。

        改进点：
        - 新增“章节数量与时长规则”与“内容边界信号”，抑制模型在长素材下回退为单章。
        - 在提示中提供总时长与期望章节数范围，提升模型对目标粒度的把握。
        - 移除“仅返回 1 章”的回退条款，避免保守合并。
        """
        # 计算可用行号范围与总数，用于提示模型严格遵循行号集合
        if meta.subtitle.items:
            _min_line_no = min(s.line_no for s in meta.subtitle.items)
            _max_line_no = max(s.line_no for s in meta.subtitle.items)
            _total_lines = len(meta.subtitle.items)
        else:
            _min_line_no = 0
            _max_line_no = 0
            _total_lines = 0

        # 计算全局时间范围与总时长（秒），用于给出“章节数与时长”的软硬约束
        if meta.subtitle.items:
            _min_sec = min(s.start_sec for s in meta.subtitle.items)
            _max_sec = max(s.end_sec for s in meta.subtitle.items)
            _total_sec = max(0.0, _max_sec - _min_sec)
        else:
            _min_sec = 0.0
            _max_sec = 0.0
            _total_sec = 0.0

        # 动态建议：基于总时长估算期望章节范围（软约束），目标粒度：每章 3~8 分钟。
        # 为避免引入 math 依赖，使用整数近似（对总秒数取整）：
        _int_total = int(_total_sec + 0.5)
        adv_min = max(3, min(20, -(-_int_total // 480)))  # ceil(T/480)
        adv_max_calc = max(3, min(20, _int_total // 180))  # floor(T/180)
        adv_max = max(adv_min, adv_max_calc)
        # 硬性下限（长素材）：≥15 分钟或 ≥120 行时，至少 4 章
        min_required = 4 if (_total_sec >= 900 or _total_lines >= 120) else 1

        sys = SystemMessage(content=dedent(
            """
            你是一名专业的结构化编辑，任务是将“给定字幕的行号列表”划分为多个“章节”。

            严格要求（必须全部满足）：
            1) 行号来源：只能使用“下面提供的字幕列表中真实出现过的行号”，禁止创造新行号、禁止估算/插值。
            2) 区间定义：章节行号区间为闭区间 [start_line_no, end_line_no]（两端均包含）。
            3) 相邻相接：相邻章节必须满足 start_line_no(i) = end_line_no(i-1) + 1。
               - 正确示例：[1-10], [11-20]
               - 错误示例：[1-10], [10-20]（共享 10 导致重叠）
            4) 全量覆盖：所有字幕行号必须被章节区间完整覆盖，无缺失、无重复、无共享行号。
            5) 无重叠且有序：任意两个章节不重叠，且按 start_line_no 升序排列。
            6) 首尾边界：第一个章节的 start_line_no 必须等于最小行号；最后一个章节的 end_line_no 必须等于最大行号。
            7) 标题生成规范：
               - 语义：标题必须基于该章节区间的实际内容进行“主题式总结”，类似书籍章节名，准确概括本章的核心主题/任务/结论。
               - 形式：使用简体中文名词短语或“主题+动作/结果”的短句；不出现“第N章/Part/Chapter”等序号。
               - 取材：优先抽取区间内的高频术语、实体名、动宾短语（如：部署脚本、索引优化、权限模型、异常处理、参数调优）。
               - 具体而不空泛：避免只给出“总结/结语/课程结尾/致谢/尾声”等泛化词，除非该区间内容本身就是对前文的系统性总结。
               - 禁止：纯英文/拼音/纯时间/纯行号/口语化寒暄（如“大家好”“谢谢观看”）/第一或第二人称（如“我们/你”）/无意义形容（如“很棒的结束”）。
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
            - 单章过长（>10 分钟）时应进一步细分，若无明显边界信号可依据主题/任务转换、术语变化、结构转折进行切分。

            内容边界信号（参考）：
            - 主题/任务/阶段切换；出现新的术语模块/子系统；从介绍 → 操作 → 结果/总结的阶段变化；
            - 提纲/转折提示词（首先/其次/然后/接着/因此/总结/回顾）；
            - 新示例/新问题/新小节开头；Q&A 段落；长时间停顿或场景显著变化（从字幕与时间上可感知）。
            """
        ))
        human = HumanMessage(content=dedent(
            f"""
            目标：将下面的完整字幕（包含行号与时间戳）按“内容导向”拆分为文章章节（非等距、非机械平均）。
            请仅基于“已出现的行号”给出章节边界，并严格遵守系统约束与输出格式。

            合法行号范围：{_min_line_no} - {_max_line_no}
            字幕总行数：{_total_lines}
            视频总时长：约 {_total_sec:.1f} 秒（≈ {_total_sec/60.0:.1f} 分）
            期望章节数范围：{adv_min} - {adv_max}（基于总时长估算）；硬性下限（若适用）：{min_required}
            注意：仅允许使用下面列表中真实出现过的行号；不得使用未出现的行号；必须全量覆盖且无重叠，排序正确。若不确定边界，请依据“内容边界信号”适度合并或拆分，并满足“章节数量与时长规则”（禁止单章回退，除非素材极短）。

            {self._render_all_subtitles(meta)}
            """
        ))
        # 业务重试：最多 3 次。每次生成后进行严格校验；失败则重试。
        max_attempts = 3
        last_err: Exception | None = None

        def _has_cjk(s: str) -> bool:
            return any('\u4e00' <= ch <= '\u9fff' for ch in s)

        for attempt in range(1, max_attempts + 1):
            t_attempt = perf_counter()
            result = self.text_llm.structured_invoke(ChaptersSchema, [sys, human], json_mode=False)

            # 标题后置中文化校验：若极端情况下仍出现非中文标题，以“第N章”占位保证中文化
            for idx, ch in enumerate(result.chapters, start=1):
                if not ch.title or not _has_cjk(ch.title):
                    ch.title = f"第{idx}章"

            try:
                # 分章边界校验：检查行号覆盖完整、无重叠、无越界
                self._validate_chapters(meta, result)
                self.logger.info(
                    "分章完成",
                    extra={
                        "chapters": [chapter.title for chapter in result.chapters],
                        "attempt": attempt,
                        "cost_ms": int((perf_counter() - t_attempt) * 1000),
                    },
                )
                return result
            except Exception as e:
                last_err = e
                self.logger.info(
                    "分章校验失败，准备重试",
                    extra={
                        "attempt": attempt,
                        "chapters": len(result.chapters),
                        "cost_ms": int((perf_counter() - t_attempt) * 1000),
                        "error": str(e),
                    },
                )
                # 下一次循环继续重试

        # 重试耗尽，抛出最后一次错误
        self.logger.error(
            "分章连续失败",
            extra={"attempts": max_attempts, "error": str(last_err) if last_err else "unknown"},
        )
        raise last_err if last_err else ValueError("分章生成连续失败")

    def _validate_chapters(self, meta: GenerationInputMeta, chs: ChaptersSchema) -> None:
        """校验分章边界结果：
        - 无重叠：任意两个章节的行号区间不重复；
        - 无缺失：章节行号并集应等于全部字幕的行号集合；
        - 无不存在行号：章节使用的行号必须都来自字幕集合；
        - 基本有效性：start_line_no ≤ end_line_no；章节按起始行号非降序（建议）。

        失败时抛出 ValueError，并给出精确原因。
        """
        # 收集字幕行号集合
        expected_lines = [s.line_no for s in meta.subtitle.items]
        if not expected_lines:
            raise ValueError("字幕为空，无法分章")
        expected_set = set(expected_lines)
        min_line = min(expected_lines)
        max_line = max(expected_lines)

        if not chs.chapters:
            raise ValueError("分章结果为空")

        # 逐章检查并收集覆盖行
        used_lines: list[int] = []
        illegal_pairs: list[tuple[int, int, str]] = []
        non_exist: set[int] = set()
        for c in chs.chapters:
            if c.start_line_no > c.end_line_no:
                illegal_pairs.append((c.start_line_no, c.end_line_no, c.title))
                continue
            # 记录首尾是否存在于字幕集合（更明确反馈）
            if c.start_line_no not in expected_set:
                non_exist.add(c.start_line_no)
            if c.end_line_no not in expected_set:
                non_exist.add(c.end_line_no)
            # 汇总该章节覆盖的行（闭区间）
            # 注意：若字幕行号连续，range 可直接覆盖；若不连续，将在“不存在行号”中体现
            used_lines.extend(range(c.start_line_no, c.end_line_no + 1))

        if illegal_pairs:
            detail = ", ".join(f"{t}:[{s}-{e}]" for s, e, t in illegal_pairs[:5])
            raise ValueError(f"发现 {len(illegal_pairs)} 个非法区间（起始>结束）：{detail}")

        used_set = set(used_lines)

        # 不存在的行号（章节引用了超出字幕集合的行）
        extra_lines = sorted(list(used_set - expected_set))
        # 缺失（章节未覆盖到的字幕行）
        missing_lines = sorted(list(expected_set - used_set))
        # 重叠（同一行在多个章节中出现）
        cnt = Counter(used_lines)
        overlapped_lines = sorted([ln for ln, n in cnt.items() if n > 1])

        # 排序与边界提示（非致命，但给出友好错误以便修复）
        sorted_by_start = sorted(chs.chapters, key=lambda x: (x.start_line_no, x.end_line_no))
        not_sorted = list(chs.chapters) != list(sorted_by_start)

        # 汇总异常并抛出（优先严格错误）
        errors: list[str] = []
        if overlapped_lines:
            sample = ", ".join(map(str, overlapped_lines[:10]))
            errors.append(f"章节行号存在重叠，样例：{sample}")
        if missing_lines:
            sample = ", ".join(map(str, missing_lines[:10]))
            errors.append(f"章节行号存在缺失，样例：{sample}")
        # 结合两种方式提示“不存在行号”（区间边界不在集合 vs 覆盖集合超出）
        non_exist_all = sorted(list(non_exist | set(extra_lines)))
        if non_exist_all:
            sample = ", ".join(map(str, non_exist_all[:10]))
            errors.append(f"章节包含不存在的行号，样例：{sample}")
        # 起止边界检查：建议覆盖完整字幕（非强制，若严格要求可改为错误）
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

    def _prompt_for_paragraphs(self, chapter_title: str, segs: List[SubtitleSegment]) -> ParagraphsSchema:
        """基于给定字幕片段生成“段落”结构。"""
        # 计算本章节的行号与时间范围，帮助模型自检
        _min_line = min(s.line_no for s in segs) if segs else 0
        _max_line = max(s.line_no for s in segs) if segs else 0
        _total_lines = len(segs)
        _min_sec = min(s.start_sec for s in segs) if segs else 0.0
        _max_sec = max(s.end_sec for s in segs) if segs else 0.0

        sys = SystemMessage(content=dedent(
            """
            你是一名专业的结构化编辑，任务是将给定章节内的字幕行划分为“段落”，并为每个段落生成 optimized。

            硬性约束（必须全部满足）：
            1) 行号一致性与连续性：仅使用输入列表中真实出现的行号；每个行号必须且仅出现一次；
               全体行号（所有段落 lines 的并集）按升序应构成从“最小行号”到“最大行号”的连续整数序列（无间断、无缺失、无重复、无额外）。
            2) 顶层时间覆盖：顶层段落按时间升序且不重叠，联合完整覆盖章节时间范围 [min_sec, max_sec]（容差 ≤ 0.05s）。
               - 相邻段落应时间相接：start_sec(i) ≈ end_sec(i-1)；不得出现空洞或交叉。
            3) 边界=行集合：任一段落 start_sec = 其 lines.start_sec 的最小值；end_sec = 其 lines.end_sec 的最大值；
               段落中每一行的 [start_sec,end_sec] 必须落在该段落边界之内（容差 ≤ 0.05s）。
            4) 排序与结构：段落按时间升序；lines 按行号升序。
            5) 输出契约（与 LangChain 结构化输出对齐）：
               - 返回格式：仅返回一个 JSON 对象（不要输出任何额外文字、注释、代码块或 Markdown），对象唯一键为 paragraphs，值为 List[段落]。
               - 段落实体验证（字段与类型）：
                 • 段落对象包含且仅包含以下字段：title, start_sec, end_sec, lines, children, optimized。
                 • 行对象包含且仅包含以下字段：line_no, start_sec, end_sec, text。
               - 体量控制与 text 规则：
                 • 为降低体量，lines.*.text 必须一律为空字符串 ""；严禁回传原文（程序将按行号本地回填）。
                 • 若底层结构化输出会省略“具有默认值”的字段，也允许省略 text，解析器将默认补 ""；但优先显式返回 text: "" 以提高一致性。
               - 字段最小化与结构：严禁输出未定义字段；children 无子段落时必须为 []；optimized 为 List[str]；lines 按行号升序。

            内省自检（输出前在你内部完成，不要打印过程）：
            - 行号校验：收集所有段落 lines 的 line_no 集合，应与输入全集完全相等；计数 Σ(唯一行) = total_lines；重复=0；缺失=0；额外=0。
            - 连续性校验：将并集按升序排序，任意相邻行号的差值恒为 1（连续数列）。
            - 时间校验：
              a) 顶层段落的时间范围按升序且不重叠；
              b) 顶层覆盖：第一个段落 start_sec ≈ min_sec；最后一个段落 end_sec ≈ max_sec；
              c) 段落边界=其 lines 的最小开始/最大结束；所有行时间落入其段落边界。

            回退策略：若任何检查未通过，则仅返回 1 个顶层段落：
            - title："整体内容概览"
            - start_sec=min_sec, end_sec=max_sec
            - lines：包含全部行（按行号升序），每行 start_sec/end_sec 与输入一致，text 一律为空字符串 ""
            - children=[]；optimized 可生成 1~N 条，保持不捏造事实

            optimized 说明：
            - 为每个段落生成 optimized（List[str]）：去除语气词、添加合理标点后，拼接成流畅句子；
            - 内容较长可自然分段；可使用 Markdown 标记重点（如 **加粗**、`行内代码`、列表等）；
            - 严禁捏造未出现在该段落字幕中的事实；保留术语与数字的准确性与时序逻辑。
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
            要求：仅可使用 S 中的行号；并且对所有段落 lines 的并集需“连续、无重复、无缺失、无额外”；
            段落之间时间不重叠且完整覆盖范围；。
            段落边界应等于其 lines 的最小开始/最大结束；若无法满足全部硬性校验，请按回退策略输出单段结构。

            {content_lines}
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
        eps = 0.05  # 容差（秒），用于吸收字幕时间边界微小重叠/空洞
        ranges = sorted([(p.start_sec, p.end_sec) for p in paras], key=lambda x: x[0])
        for i in range(1, len(ranges)):
            # 允许极小重叠（<= eps），视为相接
            if ranges[i][0] < ranges[i - 1][1] - eps:
                raise ValueError("分段时间范围存在重叠")
        min_in = min(s.start_sec for s in segs)
        max_in = max(s.end_sec for s in segs)
        if abs(ranges[0][0] - min_in) > eps or abs(ranges[-1][1] - max_in) > eps:
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
                if not (p.start_sec - eps <= l.start_sec <= l.end_sec <= p.end_sec + eps):
                    raise ValueError("段落内行时间越界")
            min_line_start = min(l.start_sec for l in p.lines)
            max_line_end = max(l.end_sec for l in p.lines)
            if abs(p.start_sec - min_line_start) > eps or abs(p.end_sec - max_line_end) > eps:
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

    def _convert_paragraph(
            self,
            ps: ParagraphSchema,
            line_lookup: Dict[int, SubtitleSegment] | None = None,
            backfill_text: bool = False,
    ) -> Paragraph:
        """将分段 Schema 转换为运行期模型。

        - 当 backfill_text=True（字幕模式）时，若行的 text 为空，则按 line_no 从 line_lookup 回填字幕原文；
        - 当 backfill_text=False（AI 笔记模式）时，保持 text 为空即可（后续不使用原文）。
        """
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
        """对分段结果进行轻量归一化，降低后续校验失败概率（最小侵入）：
        - 段落边界：start_sec/end_sec = 该段落所有行的最小开始/最大结束；
        - 行顺序：lines 按行号升序；
        - 结构顺序：同级段落按 start_sec、end_sec 升序；
        - 递归处理子层级。

        注意：不改变行集合归属，不合并/拆分段落，仅做边界与排序规范化。
        """

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

    def _process_chapter(self, ci: int, cb: ChapterBoundary, meta: GenerationInputMeta, task_out_dir: Path) -> Chapter:
        self.logger.info("处理章节", extra={"chapter_index": ci, "title": cb.title})
        segs = self._select_lines(meta.subtitle.items, cb.start_line_no, cb.end_line_no)
        if not segs:
            raise ValueError(f"章节无内容：{cb}")

        # 分段（业务重试，最多3次）
        max_attempts = 3
        pgs: ParagraphsSchema | None = None
        for attempt in range(1, max_attempts + 1):
            pgs = self._prompt_for_paragraphs(cb.title, segs)

            try:
                # 归一化边界与顺序，再进行严格校验
                self._normalize_paragraphs(pgs.paragraphs)
                self._validate_time_coverage(segs, pgs.paragraphs)
                self.logger.info(
                    "章节校验通过",
                    extra={"chapter_index": ci, "paragraphs": len(pgs.paragraphs), "attempt": attempt},
                )
                break
            except Exception:
                issue = self._analyze_coverage_issue(segs, pgs.paragraphs)
                self.logger.info(
                    "章节校验失败，准备重试",
                    extra={"chapter_index": ci, "attempt": attempt, **issue},
                )

                if attempt >= max_attempts:
                    self.logger.warning(
                        "章节校验连续失败，触发一次自动修正",
                        extra={"chapter_index": ci, "attempt": attempt, **issue},
                    )
                    try:
                        fix_report = self._auto_fix_paragraph_lines(segs, pgs.paragraphs, issue)
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
            # - 若 GUI 配置了截图目录（note.screenshot_dir），则写入该目录；
            # - 否则仍写入任务输出目录（base_dir）。
            hi_root = None
            try:
                hi_root = getattr(getattr(self.cfg, 'note', None), 'screenshot_dir', None)
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
            # 按 GUI 选择的笔记模式决定是否回填原文：
            # - 字幕模式（subtitle）：回填 text
            # - AI 笔记模式（optimized）：不回填 text，保持为空
            mode = getattr(getattr(self.cfg, "note", None), "mode", "subtitle")
            backfill = (mode == "subtitle")
            line_lookup = {s.line_no: s for s in segs}
            para = self._convert_paragraph(ps, line_lookup=line_lookup, backfill_text=backfill)
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
