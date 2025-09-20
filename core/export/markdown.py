from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.note.models import Note, Chapter, Paragraph


def _sec2hms(sec: float) -> str:
    s = int(sec)
    ms = int(round((sec - s) * 1000))
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}.{ms:03d}"


class MarkdownExporter:
    """将 Note 导出为 Markdown 文件。"""

    def __init__(self, outputs_root: Path) -> None:
        self.outputs_root = Path(outputs_root)
        self.outputs_root.mkdir(parents=True, exist_ok=True)

    def export(self, note: Note, filename: str | None = None) -> Path:
        title = note.video_path.stem
        out_file = self.outputs_root / (filename or f"{title}.md")

        lines: list[str] = []

        for ci, ch in enumerate(note.chapters, start=1):
            # 章标题提升为一级标题
            lines.append(f"# 第{ci}章 {ch.title} [{_sec2hms(ch.start_sec)} - {_sec2hms(ch.end_sec)}]")
            lines.extend(self._render_paragraphs(ci, ch.paragraphs))

        out_file.write_text("\n".join(lines), encoding="utf-8")
        return out_file

    def _render_paragraphs(self, ci: int, paragraphs: Iterable[Paragraph]) -> list[str]:
        out: list[str] = []
        for pi, p in enumerate(paragraphs, start=1):
            # 段落标题提升为二级标题
            out.append(f"## {ci}.{pi} {p.title} [{_sec2hms(p.start_sec)} - {_sec2hms(p.end_sec)}]")
            if p.image and p.image.hi_res_image_path:
                # 使用相对路径（相对于 outputs_root）
                img_path = p.image.hi_res_image_path
                try:
                    img_rel = img_path.relative_to(self.outputs_root)
                except Exception:
                    from os.path import relpath
                    img_rel = Path(relpath(img_path, start=self.outputs_root))
                out.append("")
                out.append(f"![]({img_rel.as_posix()})")
                out.append("")
            for s in p.lines:
                out.append(f"- {s.line_no} [{_sec2hms(s.start_sec)} - {_sec2hms(s.end_sec)}] {s.text}")
            out.append("")
            if p.children:
                # 子段落标题同样提升
                out.append("### 子段落")
                for ci2, c in enumerate(p.children, start=1):
                    out.append(f"- {p.title}.{ci2} [{_sec2hms(c.start_sec)} - {_sec2hms(c.end_sec)}] {c.title}")
                out.append("")
        return out
