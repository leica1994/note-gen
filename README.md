# note-gen — 基于字幕与视频的 AI 笔记生成器（PySide6 GUI）

note-gen 读取视频与同名字幕，自动完成“分章 → 分段 → 智能选图 → 导出 Markdown”，生成结构化学习笔记。发布版移除了开发期的证据归档，专注简洁与易用。

## 功能特性
- 字幕解析：支持 `.srt/.ass/.ssa/.vtt`，自动匹配同名字幕。
- 结构化生成：按时间轴分章/分段，保留原始行文本；支持“字幕模式/AI 优化模式”。
- 智能选图：等距截帧生成九宫格，调用多模态 LLM 选择代表性帧并重拍高清图。
- 稳定性：
  - 段落首尾加入时间点边缘留白 `edge_margin_sec`，减少过渡帧；
  - 子段落（三级及更深标题）同样生成图片。
- 导出与日志：生成 Markdown 与图片到 `outputs/`；运行日志在 `logs/`。

## 目录结构
```
main.py                # 入口（启动 GUI）
core/gui/app.py        # GUI 主程序
core/config/           # 配置与缓存（cache.json）
core/subtitles/        # 字幕加载
core/note/             # 分章/分段/截图/组装
core/screenshot/       # FFmpeg 截帧与九宫格
core/export/           # Markdown 导出
core/llms/             # 文本与多模态 LLM 封装
core/utils/            # 日志、哈希、重试、脱敏
```

## 环境要求
- Python 3.13+
- FFmpeg（命令 `ffmpeg` 可用，或在 GUI 中指定路径）
- 依赖管理：uv

## 安装与运行
- 安装依赖：`uv sync`
- 启动 GUI：`uv run main.py`

## 使用步骤（GUI）
1) 选择视频与同名字幕（或选择文件夹批量匹配）。
2) 配置 AI：在“AI 参数配置”中填写 `base_url`、`api_key`、`model`，可点击“测试连通性”。
3) 配置笔记参数：选择“笔记模式”；可按需设置“笔记目录/截图目录（目录）”。
4) 将候选加入任务列表，点击“开始处理任务列表”。
5) 完成后在以下位置查看输出：
   - Markdown：优先写入 GUI 配置的“笔记目录”，否则 `outputs/<视频名>.md`。
   - 图片：
     - 高清重拍：若配置了“截图目录”，则直接写到该目录；否则在 `outputs/<task_id>/...`。
     - 九宫格与缩略图：`outputs/<task_id>/chapter_x/para_y/...`。
   - 日志：`logs/<task_id>/`。

## 配置说明（cache.json）
- 文本 LLM（`text_llm`）与多模态 LLM（`mm_llm`）：
  - `base_url`、`api_key`、`model`、`concurrency`、`request_timeout`。
- 截图（`screenshot`）：
  - `ffmpeg_path`、`low_width/low_height`、`grid_columns/rows`、`hi_quality`；
  - `edge_margin_sec`：默认 `0.5` 秒，时间点避开段落首尾；
  - `max_workers`：截图并发。
- 导出（`export`）：`outputs_root`、`logs_root`。
- 笔记（`note`）：
  - `mode = subtitle | optimized`
  - `note_dir`：笔记目录（若设置，Markdown 输出写入该目录）
  - `screenshot_dir`：截图目录（若设置，高清重拍图片写入该目录）

## 生成与输出
- Markdown：优先 `<note.note_dir>/<视频名>.md`，否则 `outputs/<视频名>.md`
- 图片：
  - 高清重拍：`<note.screenshot_dir>/<视频名>_<段落标题>_<HHMMSS>.jpg`（若配置），否则 `outputs/<task_id>/...`
  - 九宫格：`outputs/<task_id>/chapter_x/para_y/grid.jpg`
- Markdown 标题：章 `#`、段 `##`、子段 `###`…；每段插入对应图片（若有）。

## 运行日志
- `logs/<task_id>/run.log`：包含加载字幕、分章/分段、九宫格与高清重拍等阶段信息（敏感信息已脱敏）。

## 故障排查
- 无法截帧：检查 FFmpeg 是否可用，或在 GUI 中设置 `ffmpeg_path`。
- LLM 错误：在“AI 参数配置”中测试连通性；检查 `base_url/api_key/model`。
- 图片卡在过渡帧：可增大 `screenshot.edge_margin_sec`（如 0.6–0.8）。

## 近期更新
- Markdown 默认文件名改为“视频名.md”。
- 高清截图默认文件名改为“视频名_段落标题_HHMMSS.jpg”（示例 11:11:11 → 111111）。
- GUI 可配置“截图目录”用于存放高清重拍图片（其他输出仍在 `outputs/`）。
- 任务列表支持双击打开“笔记目录”（未设置则打开 outputs）。
- 删除 evidence 归档功能与相关配置，简化发布。
- 子段落递归截图，解决三级标题无图问题。
- 选图提示词强化，明确拒绝“叠化/重影/运动模糊/切换覆盖层/大遮挡/截断”。
- 九宫格时间点加入边缘留白（默认 0.5s）。

## 贡献说明
- 代码风格：PEP 8、4 空格、类型注解；中文注释。
- 依赖管理：使用 `uv`；新增依赖请通过 `uv add <包名>` 并同步 `uv.lock`。
