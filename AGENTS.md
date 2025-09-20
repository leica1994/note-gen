# 仓库贡献指南

## 项目结构与模块组织
- 运行入口：`main.py`（启动 PySide6 图形界面）。
- GUI：`core/gui/app.py`；资产：`core/gui/assets/`。
- 配置与记忆：`core/config/`（`schema.py`、`cache.py`）；本地缓存：`cache.json`。
- 笔记生成：`core/note/`；字幕解析：`core/subtitles/`；导出：`core/export/`。
- 工具：`core/utils/`（日志、证据、脱敏、哈希等）；证据：`evidence/`。

## 构建、测试与本地运行
- 本仓库由 uv 管理依赖与环境；首次执行会自动创建虚拟环境。
- 安装/同步依赖（锁定）：`uv sync`
- 运行 GUI：`uv run main.py`
- 运行模块方式（可选）：`uv run -m core.gui.app`
- 新增依赖：`uv add <包名>`（自动更新 `uv.lock`）

## 代码风格与命名
- Python 3.13+，PEP 8，4 空格缩进，务必使用类型注解。
- 命名：模块/函数/变量用 `snake_case`；类用 `CamelCase`。
- 文案与注释使用中文；日志与证据中对密钥等敏感信息脱敏。
- GUI 修改遵循“最小变更边界”，非需求请勿改业务逻辑。

## 测试指南
- 框架：`pytest`（如启用）。
- 组织：`tests/` 或同目录 `test_*.py`；函数名 `test_<behavior>`。
- 运行：`uv run -m pytest -q`。优先覆盖配置读写、字幕加载、Markdown 导出等核心路径。

## Commit 与 Pull Request
- 提交信息遵循 Conventional Commits：
  - 例：`feat(gui): 候选文件支持复选框选择`
- PR 要求：目的与范围、关键变更点、手动验证步骤与截图（GUI）、关联 Issue、风险与回滚；如影响行为请更新 `evidence/`。

## 安全与配置提示
- 不要提交任何密钥；本地配置写入 `cache.json`，日志会自动脱敏。
- 产出目录：`outputs/`；日志：`logs/`；证据：`evidence/`（均需可写）。

