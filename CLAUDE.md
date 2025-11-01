# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

note-gen 是一个基于字幕与视频的 AI 笔记生成器，使用 PySide6 构建 GUI。核心功能包括：解析字幕文件（.srt/.ass/.vtt）、智能分章分段、FFmpeg 截帧、多模态 LLM 选图、导出 Markdown 笔记。

**技术栈**: Python 3.13+ | PySide6 | LangChain | FFmpeg | uv

## 核心架构

### 模块职责分离（core/ 目录）

```
core/
├── gui/              # PySide6 GUI 应用
│   └── app.py        # 主窗口、工作线程、任务队列
├── note/             # 笔记生成核心逻辑
│   ├── generator.py  # 分章分段、LLM调用、截图协调
│   └── models.py     # 数据模型（Chapter/Paragraph等）
├── llms/             # LLM 封装层
│   ├── text_llm.py   # 文本LLM（章节/段落生成）
│   └── mm_llm.py     # 多模态LLM（图片选择）
├── screenshot/       # 视频截图处理
│   ├── ffmpeg.py     # FFmpeg接口、时间点截帧
│   └── grid.py       # 九宫格时间点计算
├── subtitles/        # 字幕解析
│   ├── loader.py     # 多格式字幕加载器
│   └── models.py     # 字幕数据模型
├── export/           # Markdown 导出
│   └── markdown.py   # Markdown生成器
├── config/           # 配置管理
│   ├── cache.py      # cache.json读写
│   └── schema.py     # 配置模型
└── utils/            # 通用工具
    ├── logger.py     # 任务日志
    ├── retry.py      # 重试策略
    ├── hash.py       # 任务ID哈希
    └── secrets.py    # 敏感信息脱敏
```

### 关键数据流

1. **输入层** (`gui/app.py` → `subtitles/loader.py`)
   - GUI接收视频+字幕路径 → 加载字幕Document对象

2. **处理层** (`note/generator.py`)
   - 文本LLM分章/分段 → 生成章节/段落结构
   - 字幕时间戳 → 转换为边界点 → 计算截图时间点

3. **截图层** (`screenshot/ffmpeg.py`)
   - FFmpeg批量截帧 → 生成九宫格
   - 多模态LLM从九宫格选图 → 高清重拍

4. **输出层** (`export/markdown.py`)
   - 组装章节/段落/图片 → 生成Markdown文件

### 并发与异步

- GUI主线程：Qt界面响应
- Worker线程（QtCore.QThread）：每个任务一个线程
- ThreadPoolExecutor：批量截图（`max_workers`控制）
- 重试机制：`core/utils/retry.py`的RetryPolicy（5次重试，2s退避）

## 开发命令

### 基础命令
```bash
# 安装依赖
uv sync

# 启动GUI
uv run main.py

# 添加新依赖
uv add <package>
```

### 测试与质量（可选）
```bash
# 运行测试（pytest）
uv run -m pytest

# 代码格式化（推荐）
uv run -m black .
uv run -m isort .
```

## 关键实现要点

### 1. 配置管理（cache.json）
- 首次运行自动生成模板
- 位置：`core/config/cache.py`的`ConfigCache`
- 支持运行时修改并持久化

### 2. 任务生命周期（gui/app.py:Worker）
- 排队 → 加载字幕 → 分章分段 → 截图 → 选图 → 导出
- 进度更新：`progress_changed`信号
- 取消机制：`_cancel`标志位

### 3. 字幕解析策略
- 多格式支持：pysubs2（SRT/ASS/SSA） + webvtt-py（VTT）
- 时间转换：`subtitles/models.py`的辅助函数
- 边缘留白：`edge_margin_sec`避免过渡帧

### 4. 截图时间点计算
- 等距分布：总时长/图片数
- 边界避让：段落首尾±edge_margin_sec
- 位置：`screenshot/grid.py:generate_grid_timestamps()`

### 5. LLM调用模式
- 文本LLM：章节/段落结构化生成（JSON Schema校验）
- 多模态LLM：图片选择（九宫格→最佳帧→高清重拍）
- 并发控制：`llms/concurrency.py`（`concurrency`参数）

### 6. 图片格式与质量（2025-11-01 优化）
- **高清重拍格式**：`screenshot.hi_format`（webp/jpeg/png，默认 webp）
- **图片质量**：`screenshot.hi_quality`（webp: 1-100, jpeg: 2-31，默认10）
- **WebP优势**：同等质量下比PNG小70-80%，比JPEG小25-35%
- **配置位置**：GUI"笔记参数配置"标签页的"高清重拍图片配置"区域

### 7. 文件组织规范
```
outputs/<task_id>/           # 任务输出
├── chapter_x/para_y/        # 章节/段落目录
│   ├── grid.jpg            # 九宫格
│   └── thumbnail_*.jpg     # 缩略图
├── high_quality/            # 高清重拍（如配置）
└── <video_name>.md         # 最终笔记

logs/<task_id>/
└── run.log                  # 运行日志（敏感信息脱敏）

cache.json                   # 全局配置
```

## 扩展与维护

### 添加新字幕格式
1. 在`core/subtitles/loader.py`扩展load_subtitle函数
2. 参考pysubs2或webvtt-py的实现方式

### 添加新导出格式
1. 在`core/export/`创建新模块（如`html.py`）
2. 参考`markdown.py`的Exporter基类设计

### 修改截图逻辑
- 主要在`screenshot/ffmpeg.py`（截帧）和`grid.py`（时间点）
- 注意：`generator.py`中的ParagraphRenderTask协调流程

### LLM模型切换
- 修改`cache.json`中的`text_llm`/`mm_llm`配置
- 或在GUI的"AI参数配置"中临时修改

### 性能优化点
1. **图片体积优化**：`screenshot.hi_format=webp`（默认）→ 比PNG小70-80%
2. **图片质量平衡**：`screenshot.hi_quality=10`（默认）→ 质量与体积的折中
3. 截图并发：`screenshot.max_workers`（建议CPU核心数）
4. LLM并发：`text_llm.concurrency`（建议2-4）
5. 重试策略：`utils/retry.py`（调整次数/退避时间）

## 调试技巧

### 查看任务详情
```bash
# 任务日志
cat logs/<task_id>/run.log

# 任务输出目录
ls -R outputs/<task_id>/
```

### 常见问题定位
- **无法截帧**：检查FFmpeg路径或`cache.json.screenshot.ffmpeg_path`
- **LLM调用失败**：在GUI中点击"测试连通性"
- **过渡帧过多**：增大`screenshot.edge_margin_sec`（0.6-0.8）

### 敏感信息脱敏
- 运行时自动脱敏：`core/utils/secrets.py:mask_secret()`
- 日志中不显示API Key、base_url等

## 重要注意事项

1. **线程安全**：GUI线程只负责UI，耗时操作在Worker线程
2. **资源管理**：大文件输出到`outputs/`（不在Git追踪范围）
3. **配置持久化**：`cache.json`自动保存GUI设置
4. **错误恢复**：重试策略对所有LLM/截图异常生效
5. **中文注释**：所有代码注释使用中文（保持项目一致性）
