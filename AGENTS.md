# Repository Guidelines

## Project Structure & Module Organization
- Entry: `main.py` (launches GUI).
- Core modules: `core/` — `gui/` (app), `note/` (chapters/paragraphs), `screenshot/` (FFmpeg), `export/` (Markdown), `llms/` (text & multimodal), `config/`, `utils/`, `subtitles/`.
- Outputs & logs: `outputs/<task_id>/`, `logs/<task_id>/`.
- Config cache: `cache.json`; Python project metadata: `pyproject.toml`.

## Build, Test, and Development Commands
- Install deps: `uv sync`.
- Run locally (GUI): `uv run main.py`.
- Add dependency: `uv add <package>` (keeps `uv.lock` updated).
- FFmpeg required: ensure `ffmpeg` is on PATH or set in GUI.

## Coding Style & Naming Conventions
- Python 3.13+, PEP 8, 4-space indent, type hints required.
- Naming: `snake_case` for files/functions, `PascalCase` for classes, constants `UPPER_SNAKE_CASE`.
- Comments/docstrings in Chinese; keep functions small and cohesive. Place new modules under `core/<domain>/`.

## Testing Guidelines
- Framework: pytest (suggested). Layout: `tests/` mirrors `core/` (e.g., `tests/note/test_generator.py`).
- Naming: files `test_*.py`, functions `test_*`.
- Run: `uv run -m pytest -q`. Target: cover critical paths; prefer deterministic tests with sample subtitles.

## Commit & Pull Request Guidelines
- Commits: concise Chinese imperative summary (≤ 50 chars). Example: `优化截图选择`, `修复字幕解析` (Conventional Commits optional).
- PRs include: purpose, scope, screenshots of GUI or sample output, logs path (e.g., `logs/<task_id>/`), and any migration note. If none, state “无迁移，直接替换”。

## Security & Configuration Tips
- Do not commit secrets; set `base_url/api_key/model` in GUI. Logs mask sensitive fields.
- Large assets belong in `outputs/`; avoid tracking them in Git.
