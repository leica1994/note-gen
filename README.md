# note-gen — 基于字幕与视频的 AI 笔记生成器（PySide6 GUI）

note-gen 是一个本地运行的 AI 辅助笔记生成工具：读取视频与同名字幕，按“分章 → 分段 → 智能选图 → 导出 Markdown”的流程，生成结构化、适合知识管理的学习笔记。项目内置参数记忆与证据留存，方便审计与复现。

## 核心能力
- 字幕解析：支持 `.srt/.ass/.ssa/.vtt`，与视频同名自动配对。
- 结构化生成：LLM 识别章节与段落，保留时间轴与字幕行，支持“字幕模式/AI 优化模式”。
- 智能选图：等距截帧生成九宫格，调用多模态 LLM 选择最代表内容的截图，并重拍高清帧。
- 导出与留存：导出 Markdown；保存提示词、原始返回与过程证据到 `evidence/`；日志脱敏。

## 目录结构
```
main.py                # 应用入口（启动 GUI）
core/gui/app.py        # GUI 主程序（左右布局 + 参数记忆）
core/gui/assets/       # 图标与界面资源
core/config/           # 配置模型与 cache（cache.json）
core/subtitles/        # 字幕加载与模型
core/note/             # 笔记生成（分章/分段/选图）
core/screenshot/       # ffmpeg 截图与九宫格
core/export/           # Markdown 导出
core/utils/            # 证据、日志、哈希、脱敏、重试
```

## 环境与安装（uv 管理）
- 依赖：Python 3.13+、FFmpeg（命令 `ffmpeg` 可用，或在 GUI 中调整路径）。
- 安装与同步：`uv sync`
- 启动 GUI：`uv run main.py`
- 以模块方式运行（可选）：`uv run -m core.gui.app`
- 新增依赖：`uv add <包名>`（自动更新 `uv.lock`）

## 截图稳定性改进（2025-09-21）
- 新增 `ScreenshotConfig.edge_margin_sec`（默认 0.5s）：九宫格时间点在计算时自动避开段落首尾的过渡帧（淡入/淡出），降低“卡在过渡帧”的概率。
- 为子段落（三级及更深标题）同样生成九宫格与高清截图；Markdown 递归渲染图片，弥补此前子段无图的问题。
- 如需恢复旧行为，将 `edge_margin_sec` 设为 `0.0` 即可。

## GUI 使用说明（新版布局）
- 左右布局：右侧“任务列表”；左侧上下分为“选择与候选列表”“参数配置”。
- 左上 – 选择与候选列表：
  - 选择模式：
    - 文件模式：选择一个视频，自动匹配同名字幕。
    - 文件夹模式：扫描目录，查找同名“视频-字幕”对（当前不递归）。
  - 候选表：列为【选择/视频/字幕/目录】，支持复选框、多选；提供“全选/全不选/添加到右侧任务列表”。
- 左下 – 参数配置（标签页）：
  - AI 参数配置：文本 LLM 与多模态 LLM 的 `base_url/api_key/model/并发`，支持“连通性测试”。
  - 笔记参数配置：`note.mode`（`subtitle`/`optimized`）。所有参数变更自动写入 `cache.json`。
- 右侧 – 任务列表：
  - 列为【任务ID/视频/字幕/状态/进度%/错误】；选中后点击“开始处理选中任务”。

## 处理流程与产物
1) 加载字幕：`core/subtitles/loader.py`
2) 生成笔记：`core/note/generator.py`
   - 分章/分段：`TextLLM` 结构化输出（见 `core/llms/`）。
   - 选图：`core/screenshot/` 生成九宫格，`MultiModalLLM` 选择，随后高清重拍。
3) 导出 Markdown：`core/export/markdown.py`（文件名：`outputs/<task_id>/<task_id>.md`）
4) 证据与日志：
   - `evidence/<task_id>/` 保存提示词、原始返回与中间结果（`EvidenceWriter`）。
   - `logs/` 记录任务全流程（敏感信息经 `mask_secret` 脱敏）。

## 参数要点（GUI 可视化配置）
- 文本 LLM/多模态 LLM：`base_url`、`api_key`、`model`、`concurrency`、`timeout` 等。
- 截图：`ffmpeg_path`、`grid_columns/rows`、`hi_quality`、`max_workers`。
- 导出：`outputs_root`、`evidence_root`、`logs_root`、`save_prompts_and_raw`。
- 笔记模式：`note.mode = subtitle | optimized`。

## 常见问题
- 无法截帧：确认已安装 FFmpeg 或修改 GUI 中的 `ffmpeg_path`。
- LLM 连通性报错：在 GUI “AI 参数配置”中使用“测试连通性”自检；检查 `base_url/api_key/model`。
- 未匹配到字幕：确保字幕与视频同名，扩展名为 `.srt/.ass/.ssa/.vtt`。

## 贡献与说明
- 开发规范与提交要求见 `AGENTS.md`。
- 建议使用 Conventional Commits；GUI 改动注意“最小变更边界”，并在 `evidence/` 留存关键证据。
