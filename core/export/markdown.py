from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.note.models import Note, Paragraph


class MarkdownExporter:
    """将 Note 导出为 Markdown 文件。"""

    def __init__(self, outputs_root: Path, note_mode: str = "subtitle") -> None:
        self.outputs_root = Path(outputs_root)
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        # 笔记模式：subtitle / optimized
        self.note_mode = note_mode

    def export(self, note: Note, filename: str | None = None) -> Path:
        title = note.video_path.stem
        out_file = self.outputs_root / (filename or f"{title}.md")

        lines: list[str] = []

        for ci, ch in enumerate(note.chapters, start=1):
            # 章标题为一级标题
            lines.append(f"# 第{ci}章 {ch.title}")
            lines.extend(self._render_paragraphs(ci, ch.paragraphs))

        out_file.write_text("\n".join(lines), encoding="utf-8")
        return out_file

    def _render_paragraphs(self, ci: int, paragraphs: Iterable[Paragraph]) -> list[str]:
        """渲染一章内的所有段落（含递归子段）。

        规则：
        - 顶层段落使用二级标题（##）。
        - 子段落标题相较父级增加 1 个 #（最多不超过 6）。
        - 标题格式统一：`<#...#> <编号> <标题> [<start> - <end>]`。
        - 每个段落输出其图片（若有）与行列表；随后递归输出其子段。
        """
        out: list[str] = []
        for pi, p in enumerate(paragraphs, start=1):
            idx_prefix = f"{ci}.{pi}"
            out.extend(self._render_single_paragraph(heading_level=2, idx_prefix=idx_prefix, p=p))
        return out

    def _render_single_paragraph(self, heading_level: int, idx_prefix: str, p: Paragraph) -> list[str]:
        """递归渲染单个段落及其子段。

        参数：
        - heading_level: 标题层级（2 表示顶层段落使用 ##）。
        - idx_prefix: 编号前缀，例如 "1.2"、"1.2.3"。
        - p: 段落对象。
        """
        out: list[str] = []

        # 生成标题行：根据层级生成对应数量的 '#'
        level = max(1, min(6, heading_level))
        hashes = "#" * level
        out.append(f"{hashes} {idx_prefix} {p.title}")

        # 图片（若有）：使用相对路径（相对于 outputs_root）
        if p.image and p.image.hi_res_image_path:
            img_path = p.image.hi_res_image_path
            # 尝试生成相对路径；跨盘符（如 C: 与 D:）时退回为绝对路径
            try:
                img_rel = img_path.relative_to(self.outputs_root)
            except Exception:
                from os.path import relpath
                try:
                    img_rel = Path(relpath(str(img_path), start=str(self.outputs_root)))
                except Exception:
                    # 跨盘符或其他异常：直接使用绝对路径，Markdown 解析器通常可识别本地绝对路径
                    img_rel = img_path
            out.append("")
            out.append(f"![]({img_rel.as_posix()})")
            out.append("")

        # 段落内容：按模式输出
        if self.note_mode == "optimized" and p.optimized:
            # 直接输出优化后的文本（可含 Markdown 标记）
            for item in p.optimized:
                out.append(item)
                out.append("")
        else:
            # 字幕模式（逐行）
            for s in p.lines:
                out.append(f"- {s.text}")
            out.append("")

        # 递归子段：层级 +1，编号前缀追加
        if p.children:
            for ci2, c in enumerate(p.children, start=1):
                child_prefix = f"{idx_prefix}.{ci2}"
                out.extend(self._render_single_paragraph(heading_level=level + 1, idx_prefix=child_prefix, p=c))

        return out
