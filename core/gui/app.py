from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

from PySide6 import QtCore, QtWidgets, QtGui

from core.config.cache import ConfigCache
from core.config.schema import AppConfig
from core.export.markdown import MarkdownExporter
from core.note.generator import NoteGenerator
from core.note.models import GenerationInputMeta
from core.subtitles.loader import load_subtitle
from core.utils.hash import hash_task
from core.utils.logger import init_task_logger
from core.utils.secrets import mask_secret


@dataclass
class TaskItem:
    video: Path
    subtitle: Path
    status: str = "排队"
    progress: int = 0
    task_id: Optional[str] = None
    error: Optional[str] = None


class Worker(QtCore.QThread):
    progress_changed = QtCore.Signal(int)
    status_changed = QtCore.Signal(str)
    finished_with_result = QtCore.Signal(str, str)  # (status, error)

    def __init__(self, cfg: AppConfig, task: TaskItem, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.task = task

    def run(self):
        try:
            self.status_changed.emit("进行中")
            # 0) 参数快照与任务ID
            params = json.loads(self.cfg.model_dump_json())
            self.task.task_id = hash_task(self.task.video, self.task.subtitle, params)

            # 输出目录：任务工作目录维持在项目根 outputs 下；Markdown 根目录遵循 GUI 笔记目录（若配置）。
            out_dir = Path(self.cfg.export.outputs_root) / (self.task.task_id or "unknown")
            out_dir.mkdir(parents=True, exist_ok=True)

            # 初始化任务日志
            logger = init_task_logger(self.task.task_id or "unknown", self.cfg.export.logs_root)
            logger.info("任务开始", extra={
                "video": str(self.task.video),
                "subtitle": str(self.task.subtitle),
            })
            # 记录输出目录位置（便于审计）
            # 计算 Markdown 根目录（若配置了 note.note_dir 则优先）
            try:
                md_root = Path(getattr(getattr(self.cfg, 'note', None), 'note_dir', None) or self.cfg.export.outputs_root)
            except Exception:
                md_root = Path(self.cfg.export.outputs_root)
            try:
                logger.info("输出目录", extra={
                    "markdown_root": str(md_root),
                    "screenshots_task_dir": str(out_dir),
                    "hi_image_override_dir": str(getattr(getattr(self.cfg, 'note', None), 'screenshot_dir', '') or ''),
                })
            except Exception:
                pass
            # 记录配置（敏感字段打码）
            logger.info("配置快照", extra={
                "text_llm": {
                    "base_url": self.cfg.text_llm.base_url,
                    "api_key": mask_secret(self.cfg.text_llm.api_key),
                    "model": self.cfg.text_llm.model,
                    "temperature": self.cfg.text_llm.temperature,
                },
                "mm_llm": {
                    "base_url": self.cfg.mm_llm.base_url,
                    "api_key": mask_secret(self.cfg.mm_llm.api_key),
                    "model": self.cfg.mm_llm.model,
                },
                "screenshot": self.cfg.screenshot.model_dump(mode="json"),
                "export": self.cfg.export.model_dump(mode="json"),
                "note": getattr(self.cfg, 'note', None).model_dump(mode="json") if getattr(self.cfg, 'note',
                                                                                           None) else {
                    "mode": "subtitle"},
            })

            self.progress_changed.emit(5)
            # 1) 加载字幕
            t0 = perf_counter()
            sub_doc = load_subtitle(self.task.subtitle)
            logger.info("字幕加载完成", extra={
                "segments": len(sub_doc.items),
                "format": sub_doc.format,
                "cost_ms": int((perf_counter() - t0) * 1000),
            })
            self.progress_changed.emit(15)
            # 2) 生成笔记
            generator = NoteGenerator(self.cfg, logger=logger)
            meta = GenerationInputMeta(video_path=self.task.video, subtitle=sub_doc, params=params)
            t1 = perf_counter()
            note = generator.generate(meta, out_dir)
            logger.info("笔记生成完成", extra={
                "chapters": len(note.chapters),
                "cost_ms": int((perf_counter() - t1) * 1000),
            })
            self.progress_changed.emit(85)
            # 3) 导出 Markdown
            exporter = MarkdownExporter(md_root, note_mode=(
                self.cfg.note.mode if getattr(self.cfg, 'note', None) else 'subtitle'))
            # 变更：默认笔记文件名改为“视频名称.md”（不再使用任务ID）
            md = exporter.export(note)
            logger.info("导出 Markdown 完成", extra={"path": str(md)})
            self.progress_changed.emit(100)
            logger.info("任务结束", extra={"status": "成功"})
            self.finished_with_result.emit("成功", "")
        except Exception as e:  # noqa: BLE001
            try:
                logger = init_task_logger(self.task.task_id or "unknown", self.cfg.export.logs_root)
                logger.error("任务失败", extra={"error": str(e)})
            except Exception:
                pass
            self.finished_with_result.emit("失败", str(e))


# 取消自绘委托，改为 cellWidget 方式实现复选框居中显示


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("note-gen：AI笔记")
        self.resize(1200, 720)
        # 应用图标（窗口图标）
        try:
            self.setWindowIcon(self._icon("app"))
        except Exception:
            pass

        self.cache = ConfigCache()
        self.cfg = self.cache.load()

        # 任务与候选文件对
        self.current_worker: Optional[Worker] = None
        self.tasks: list[TaskItem] = []
        # 顺序执行队列：保存“排队”状态任务的行索引，点击开始后依次串行处理
        self.pending_queue: list[int] = []
        self.candidates: list[tuple[Path, Path]] = []  # 左侧候选“视频-字幕”对（仅界面列表，不触发处理）

        self._build_ui()
        self._connect_signals()
        self._apply_styles()

        # 自动保存防抖计时器
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)  # ms
        self._save_timer.timeout.connect(self._save_config)

    def _build_ui(self):
        """
        重构布局（仅GUI样式与组织，不改变参数与处理逻辑）：
        - 整体左右布局：右侧为任务列表；左侧再拆分为上下布局。
        - 左上：文件/文件夹选择 + 候选文件对列表 + “添加到右侧任务列表”。
        - 左下：参数配置，分为两个标签页（AI参数配置、笔记参数配置）。
        """
        # 顶层水平分割：左（输入与参数） | 右（任务列表）
        root_splitter = QtWidgets.QSplitter()
        root_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(root_splitter)

        # 左侧再用垂直分割：上（文件与候选） | 下（参数Tabs）
        left_splitter = QtWidgets.QSplitter()
        left_splitter.setOrientation(QtCore.Qt.Orientation.Vertical)

        # ============ 左上：文件选择与候选列表 ============
        top_left = QtWidgets.QWidget()
        top_left_layout = QtWidgets.QVBoxLayout(top_left)
        top_left_layout.setContentsMargins(6, 6, 6, 6)
        top_left_layout.setSpacing(6)

        # 模式切换行（文件模式 / 文件夹模式）
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("选择模式："))
        self.radio_mode_file = QtWidgets.QRadioButton("文件模式")
        self.radio_mode_folder = QtWidgets.QRadioButton("文件夹模式")
        self.radio_mode_file.setChecked(True)
        mode_row.addWidget(self.radio_mode_file)
        mode_row.addWidget(self.radio_mode_folder)
        mode_row.addStretch(1)
        top_left_layout.addLayout(mode_row)

        # 选择按钮
        self.btn_pick_input = QtWidgets.QPushButton()
        self._update_pick_button_text()
        top_left_layout.addWidget(self.btn_pick_input)

        # 候选文件对列表（选择、视频、字幕、目录）
        self.file_table = QtWidgets.QTableWidget(0, 4)
        self.file_table.setHorizontalHeaderLabels(["选择", "视频", "字幕", "目录"])
        self.file_table.horizontalHeader().setStretchLastSection(True)
        self.file_table.verticalHeader().setVisible(False)
        # 使用复选框，不使用行选择
        self.file_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.file_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        # 将“选择”列设置为固定窄宽以贴合复选框显示
        try:
            self.file_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.file_table.setColumnWidth(0, 56)
            self.file_table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        except Exception:
            pass
        top_left_layout.addWidget(self.file_table)

        # 候选选择控制区：全选 / 全不选 / 清空 / 添加到任务
        cand_btn_row = QtWidgets.QHBoxLayout()
        self.btn_select_all_cand = QtWidgets.QPushButton("全选")
        self.btn_deselect_all_cand = QtWidgets.QPushButton("全不选")
        self.btn_clear_cand = QtWidgets.QPushButton("清空")
        self.btn_add_task = QtWidgets.QPushButton("添加到任务列表")
        cand_btn_row.addWidget(self.btn_select_all_cand)
        cand_btn_row.addWidget(self.btn_deselect_all_cand)
        cand_btn_row.addStretch(1)
        cand_btn_row.addWidget(self.btn_clear_cand)
        cand_btn_row.addWidget(self.btn_add_task)
        top_left_layout.addLayout(cand_btn_row)

        # ============ 左下：参数Tabs ============
        bottom_tabs = QtWidgets.QTabWidget()

        # --- AI参数配置 ---
        tab_ai = QtWidgets.QWidget()
        cfg_layout = QtWidgets.QFormLayout(tab_ai)
        self.edit_base_url = QtWidgets.QLineEdit(self.cfg.text_llm.base_url or "")
        self.edit_api_key = QtWidgets.QLineEdit(self.cfg.text_llm.api_key or "")
        self.edit_api_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.edit_model = QtWidgets.QLineEdit(self.cfg.text_llm.model)
        self.spin_text_conc = QtWidgets.QSpinBox()
        # 放开上限：仅设置最小值为 1，最大值使用 32 位整型上限
        self.spin_text_conc.setRange(1, 2147483647)
        self.spin_text_conc.setValue(self.cfg.text_llm.concurrency)
        self.btn_test_llm = QtWidgets.QPushButton("测试文本LLM连通性")

        self.edit_mm_base_url = QtWidgets.QLineEdit(self.cfg.mm_llm.base_url or "")
        self.edit_mm_api_key = QtWidgets.QLineEdit(self.cfg.mm_llm.api_key or "")
        self.edit_mm_api_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.edit_mm_model = QtWidgets.QLineEdit(self.cfg.mm_llm.model)
        self.spin_mm_conc = QtWidgets.QSpinBox()
        # 放开上限：仅设置最小值为 1，最大值使用 32 位整型上限
        self.spin_mm_conc.setRange(1, 2147483647)
        self.spin_mm_conc.setValue(self.cfg.mm_llm.concurrency)
        self.btn_test_mm = QtWidgets.QPushButton("测试多模态LLM连通性")

        cfg_layout.addRow("文本LLM base_url", self.edit_base_url)
        cfg_layout.addRow("文本LLM api_key", self.edit_api_key)
        cfg_layout.addRow("文本LLM model", self.edit_model)
        cfg_layout.addRow("文本LLM并发", self.spin_text_conc)
        cfg_layout.addRow(self.btn_test_llm)
        cfg_layout.addRow(QtWidgets.QLabel("——"))
        cfg_layout.addRow("多模态 base_url", self.edit_mm_base_url)
        cfg_layout.addRow("多模态 api_key", self.edit_mm_api_key)
        cfg_layout.addRow("多模态 model", self.edit_mm_model)
        cfg_layout.addRow("多模态并发", self.spin_mm_conc)
        cfg_layout.addRow(self.btn_test_mm)

        bottom_tabs.addTab(tab_ai, "AI参数配置")

        # --- 笔记参数配置 ---
        tab_note = QtWidgets.QWidget()
        note_layout = QtWidgets.QFormLayout(tab_note)
        self.combo_note_mode = QtWidgets.QComboBox()
        self.combo_note_mode.addItem("字幕模式", userData="subtitle")
        self.combo_note_mode.addItem("AI优化模式", userData="optimized")
        try:
            current_mode = (self.cfg.note.mode if getattr(self.cfg, 'note', None) else 'subtitle') or 'subtitle'
        except Exception:
            current_mode = 'subtitle'
        idx = 0 if current_mode == 'subtitle' else 1
        self.combo_note_mode.setCurrentIndex(idx)
        note_layout.addRow("笔记模式", self.combo_note_mode)

        # 新增：笔记目录（用于 Markdown 输出根目录）
        self.edit_note_dir = QtWidgets.QLineEdit()
        try:
            self.edit_note_dir.setText(str(getattr(self.cfg.note, 'note_dir', '') or ''))
        except Exception:
            self.edit_note_dir.setText("")
        self.btn_pick_note_dir = QtWidgets.QPushButton("选择目录")
        row_note_input = QtWidgets.QWidget()
        row_note_input_layout = QtWidgets.QHBoxLayout(row_note_input)
        row_note_input_layout.setContentsMargins(0, 0, 0, 0)
        row_note_input_layout.setSpacing(6)
        row_note_input_layout.addWidget(self.edit_note_dir)
        row_note_input_layout.addWidget(self.btn_pick_note_dir)
        note_layout.addRow("笔记目录", row_note_input)

        # 新增：截图目录（用于高清重拍输出目录）
        self.edit_screenshot_dir = QtWidgets.QLineEdit()
        try:
            self.edit_screenshot_dir.setText(str(getattr(self.cfg.note, 'screenshot_dir', '') or ''))
        except Exception:
            self.edit_screenshot_dir.setText("")
        self.btn_pick_screenshot_dir = QtWidgets.QPushButton("选择目录")
        row_shot_input = QtWidgets.QWidget()
        row_shot_input_layout = QtWidgets.QHBoxLayout(row_shot_input)
        row_shot_input_layout.setContentsMargins(0, 0, 0, 0)
        row_shot_input_layout.setSpacing(6)
        row_shot_input_layout.addWidget(self.edit_screenshot_dir)
        row_shot_input_layout.addWidget(self.btn_pick_screenshot_dir)
        note_layout.addRow("截图目录", row_shot_input)
        bottom_tabs.addTab(tab_note, "笔记参数配置")

        # 左侧组装
        left_splitter.addWidget(top_left)
        left_splitter.addWidget(bottom_tabs)
        left_splitter.setSizes([450, 270])

        # ============ 右侧：任务列表 ============
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        # 右侧任务表：将“错误”列改为“操作”列（内含删除按钮）
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["任务ID", "视频", "字幕", "状态", "进度%", "操作"])
        # 调整列宽策略：操作列固定窄宽，其余列合理伸缩
        try:
            header = self.table.horizontalHeader()
            header.setStretchLastSection(False)
            # 操作列（第5列）固定宽度，避免被拉伸
            header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(5, 80)
            # 视频与字幕列自适应填充
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
            # 表头文本居中
            header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        except Exception:
            pass
        self.table.verticalHeader().setVisible(False)
        # 更新按钮文案以匹配新行为：执行任务列表中所有“排队”任务
        self.btn_start = QtWidgets.QPushButton("开始处理任务列表")
        right_layout.addWidget(self.table)
        right_layout.addWidget(self.btn_start)

        # 顶层装配
        root_splitter.addWidget(left_splitter)
        root_splitter.addWidget(right)
        root_splitter.setSizes([520, 680])

        # 统一为所有主要按钮设置默认图标（避免首次操作前未显示图标的情况）
        try:
            self._apply_default_icons()
        except Exception:
            pass

        # 放开并发上限：在构建时已设置为 1..2147483647（仅最小值限制）

    def _connect_signals(self):
        # 左侧：模式切换与选择输入
        self.radio_mode_file.toggled.connect(self._update_pick_button_text)
        self.radio_mode_folder.toggled.connect(self._update_pick_button_text)
        self.btn_pick_input.clicked.connect(self._pick_input)
        self.btn_add_task.clicked.connect(self._add_task)
        self.btn_select_all_cand.clicked.connect(self._select_all_candidates)
        self.btn_deselect_all_cand.clicked.connect(self._deselect_all_candidates)
        self.btn_clear_cand.clicked.connect(self._clear_candidates)
        # 修改逻辑：点击后顺序执行所有“排队”任务，而非仅执行选中项
        self.btn_start.clicked.connect(self._start_all_pending)
        self.btn_test_llm.clicked.connect(self._test_text_llm)
        self.btn_test_mm.clicked.connect(self._test_mm_llm)
        # 参数变更自动保存（防抖）
        self.edit_base_url.textChanged.connect(self._schedule_save_config)
        self.edit_api_key.textChanged.connect(self._schedule_save_config)
        self.edit_model.textChanged.connect(self._schedule_save_config)
        self.edit_mm_base_url.textChanged.connect(self._schedule_save_config)
        self.edit_mm_api_key.textChanged.connect(self._schedule_save_config)
        self.edit_mm_model.textChanged.connect(self._schedule_save_config)
        self.spin_text_conc.valueChanged.connect(self._schedule_save_config)
        self.spin_mm_conc.valueChanged.connect(self._schedule_save_config)
        # 笔记模式变更
        self.combo_note_mode.currentIndexChanged.connect(self._schedule_save_config)
        # 新增：笔记/截图目录变更与选择
        self.edit_note_dir.textChanged.connect(self._schedule_save_config)
        self.edit_screenshot_dir.textChanged.connect(self._schedule_save_config)
        self.btn_pick_note_dir.clicked.connect(self._pick_note_dir)
        self.btn_pick_screenshot_dir.clicked.connect(self._pick_screenshot_dir)
        # 右侧任务表：双击打开输出目录（笔记目录优先）
        try:
            self.table.cellDoubleClicked.connect(self._on_table_double_click)
        except Exception:
            pass

    # 左侧交互
    def _update_pick_button_text(self):
        """根据模式更新选择按钮文案。"""
        if getattr(self, "btn_pick_input", None) is None:
            return
        if self.radio_mode_file.isChecked():
            self.btn_pick_input.setText("选择视频文件")
            self.btn_pick_input.setIcon(self._icon("file"))
        else:
            self.btn_pick_input.setText("选择文件夹")
            self.btn_pick_input.setIcon(self._icon("folder"))

    def _apply_default_icons(self):
        """为界面中的主要按钮设置默认图标（一次性）。

        说明：此前部分图标在特定交互后才被设置，导致首次进入时按钮无图标。
        这里统一在 UI 构建后设置，保证默认就有图标显示。
        """
        # 选择输入按钮（受模式影响，复用已有逻辑）
        self._update_pick_button_text()
        # 候选区控制按钮
        if getattr(self, "btn_add_task", None):
            self.btn_add_task.setIcon(self._icon("add"))
        if getattr(self, "btn_select_all_cand", None):
            self.btn_select_all_cand.setIcon(self._icon("add"))
        if getattr(self, "btn_deselect_all_cand", None):
            self.btn_deselect_all_cand.setIcon(self._icon("minus"))
        # AI 连通性测试
        if getattr(self, "btn_test_llm", None):
            self.btn_test_llm.setIcon(self._icon("robot"))
        if getattr(self, "btn_test_mm", None):
            self.btn_test_mm.setIcon(self._icon("camera"))
        # 任务执行按钮
        if getattr(self, "btn_start", None):
            self.btn_start.setIcon(self._icon("play"))

    def _pick_input(self):
        """根据模式选择视频或文件夹并填充候选文件对。"""
        if self.radio_mode_file.isChecked():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频", "",
                                                            "视频文件 (*.mp4 *.mkv *.mov *.avi *.wmv)")
            if not path:
                return
            video = Path(path)
            sub = self._auto_match_subtitle(video)
            if not sub:
                QtWidgets.QMessageBox.warning(self, "未找到字幕", f"未找到与 {video.name} 同名的字幕文件")
                return
            self._append_candidates([(video, sub)])
        else:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择包含视频与字幕的文件夹", "")
            if not folder:
                return
            pairs = self._scan_folder_for_pairs(Path(folder))
            if not pairs:
                QtWidgets.QMessageBox.information(self, "没有匹配", "未在该目录下找到同名的视频-字幕对")
                return
            self._append_candidates(pairs)

    def _append_candidates(self, pairs: list[tuple[Path, Path]]):
        """将新发现的候选对追加到内存并刷新表格（去重）。"""
        existed = {(v.resolve(), s.resolve()) for v, s in self.candidates}
        for v, s in pairs:
            key = (v.resolve(), s.resolve())
            if key not in existed:
                self.candidates.append((v, s))
                existed.add(key)
        self._refresh_candidates_table()
        # 为添加与批量选择按钮设置图标（仅样式增强）
        try:
            self.btn_add_task.setIcon(self._icon("add"))
            self.btn_select_all_cand.setIcon(self._icon("add"))
            self.btn_deselect_all_cand.setIcon(self._icon("minus"))
        except Exception:
            pass

    def _refresh_candidates_table(self):
        self.file_table.setRowCount(len(self.candidates))
        for i, (v, s) in enumerate(self.candidates):
            # 使用 cellWidget + 居中布局放置复选框
            cb_container = QtWidgets.QWidget()
            cb_layout = QtWidgets.QHBoxLayout(cb_container)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            cb_layout.setSpacing(0)
            cb = QtWidgets.QCheckBox()
            cb.setObjectName("candCheck")
            cb_layout.addStretch(1)
            cb_layout.addWidget(cb)
            cb_layout.addStretch(1)
            self.file_table.setCellWidget(i, 0, cb_container)

            def cell(txt: str):
                item = QtWidgets.QTableWidgetItem(txt)
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                return item

            self.file_table.setItem(i, 1, cell(v.name))
            self.file_table.setItem(i, 2, cell(s.name))
            self.file_table.setItem(i, 3, cell(str(v.parent)))

    def _add_task(self):
        """将左侧候选表中勾选的行添加到右侧任务列表。"""
        any_checked = False
        for row in range(self.file_table.rowCount()):
            cw = self.file_table.cellWidget(row, 0)
            if not cw:
                continue
            cb = cw.findChild(QtWidgets.QCheckBox, "candCheck")
            if cb and cb.isChecked():
                any_checked = True
                try:
                    v, s = self.candidates[row]
                    self.tasks.append(TaskItem(video=v, subtitle=s))
                except Exception:
                    continue
        if not any_checked:
            QtWidgets.QMessageBox.information(self, "提示", "请先勾选左侧候选列表中的一行或多行")
            return
        # 1) 刷新右侧任务表
        self._refresh_table()
        # 2) 需求变更：点击“添加到任务列表”后，应清空左侧文件列表
        #    说明：这里直接清空候选数据源并刷新候选表格，以满足“添加后清空”的交互期望。
        try:
            self.candidates.clear()
            self._refresh_candidates_table()
        except Exception:
            # 兜底：若刷新过程中出现异常，至少将表格行数置零，避免残留展示
            try:
                self.file_table.setRowCount(0)
            except Exception:
                pass

    # 文件匹配与扫描
    def _auto_match_subtitle(self, video: Path) -> Optional[Path]:
        """在同目录下按文件名（不含扩展名）自动匹配字幕。"""
        sub_exts = [".srt", ".ass", ".ssa", ".vtt"]
        stem = video.stem
        for ext in sub_exts:
            cand = video.with_name(stem + ext)
            if cand.exists():
                return cand
        # 大小写或其他扩展名变体（宽松匹配）
        try:
            for p in video.parent.iterdir():
                if p.is_file() and p.stem == stem and p.suffix.lower() in sub_exts:
                    return p
        except Exception:
            pass
        return None

    def _scan_folder_for_pairs(self, folder: Path) -> list[tuple[Path, Path]]:
        """扫描文件夹，查找同名“视频-字幕”对（仅当前目录，最小变更）。"""
        video_exts = {".mp4", ".mkv", ".mov", ".avi", ".wmv"}
        sub_exts = {".srt", ".ass", ".ssa", ".vtt"}
        videos: dict[str, Path] = {}
        subs: dict[str, Path] = {}
        try:
            for p in folder.iterdir():
                if not p.is_file():
                    continue
                suf = p.suffix.lower()
                if suf in video_exts:
                    videos[p.stem] = p
                elif suf in sub_exts:
                    subs[p.stem] = p
        except Exception:
            return []
        pairs: list[tuple[Path, Path]] = []
        for stem, v in videos.items():
            s = subs.get(stem)
            if s:
                pairs.append((v, s))
        return pairs

    # 右侧交互
    def _refresh_table(self):
        self.table.setRowCount(len(self.tasks))
        for i, t in enumerate(self.tasks):
            def cell(txt: str):
                item = QtWidgets.QTableWidgetItem(txt)
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                return item

            self.table.setItem(i, 0, cell(t.task_id or "-"))
            self.table.setItem(i, 1, cell(t.video.name))
            self.table.setItem(i, 2, cell(t.subtitle.name))
            self.table.setItem(i, 3, cell(t.status))
            self.table.setItem(i, 4, cell(str(t.progress)))
            # 操作列：删除按钮，点击后从任务列表移除该行
            action_container = QtWidgets.QWidget()
            action_layout = QtWidgets.QHBoxLayout(action_container)
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(0)
            action_layout.addStretch(1)
            btn_del = QtWidgets.QPushButton("删除")
            try:
                btn_del.setIcon(self._icon("trash"))
            except Exception:
                pass
            # 捕获当前 i 作为行索引
            btn_del.clicked.connect(lambda _=False, row=i: self._delete_task_row(row))
            action_layout.addWidget(btn_del)
            action_layout.addStretch(1)
            self.table.setCellWidget(i, 5, action_container)
        # 按钮图标（样式增强，不影响逻辑）
        try:
            self.btn_start.setIcon(self._icon("play"))
            self.btn_test_llm.setIcon(self._icon("robot"))
            self.btn_test_mm.setIcon(self._icon("camera"))
            if getattr(self, "btn_clear_cand", None):
                self.btn_clear_cand.setIcon(self._icon("trash"))
        except Exception:
            pass

    def _clear_candidates(self):
        """清空左侧候选文件列表。"""
        try:
            self.candidates.clear()
            self._refresh_candidates_table()
        except Exception:
            try:
                self.file_table.setRowCount(0)
            except Exception:
                pass

    def _delete_task_row(self, row: int):
        """从任务列表中删除指定行。

        - 若该任务正在执行，则阻止删除并提示；
        - 删除后重建排队队列并刷新表格。
        """
        if row < 0 or row >= len(self.tasks):
            return
        # 若正在执行当前行任务，阻止删除
        try:
            if self.current_worker and self.current_worker.isRunning():
                if self.tasks[row] is getattr(self.current_worker, 'task', None):
                    QtWidgets.QMessageBox.information(self, "提示", "当前任务进行中，无法删除")
                    return
        except Exception:
            pass
        try:
            del self.tasks[row]
        except Exception:
            return
        # 删除后重建“排队”队列并刷新
        try:
            self.pending_queue = [i for i, t in enumerate(self.tasks) if (t.status or "").strip() == "排队"]
        except Exception:
            self.pending_queue = []
        self._refresh_table()

    def _start_row(self, row: int):
        """启动指定行的任务（串行执行的原子操作）。

        保持最小变更：复用原有选中项启动逻辑，仅将行索引作为参数传入。
        """
        if row < 0 or row >= len(self.tasks):
            return
        if self.current_worker and self.current_worker.isRunning():
            # 已有任务进行中，由队列驱动下一个，不在此重复启动
            return
        task = self.tasks[row]
        # 启动前保存一次最新配置
        self._save_config()

        worker = Worker(self.cfg, task)
        self.current_worker = worker
        worker.progress_changed.connect(lambda p: self._on_progress(row, p))
        worker.status_changed.connect(lambda s: self._on_status(row, s))
        worker.finished_with_result.connect(lambda st, err: self._on_finish(row, st, err))
        worker.start()

    def _start_all_pending(self):
        """收集并顺序执行所有“排队”任务。

        - 若当前有任务在执行：仅重置/填充队列，等待当前完成后自动继续；
        - 若无任务在执行：立即启动队列第一个并串行推进。
        """
        # 收集当前“排队”任务的行索引，按列表顺序执行
        self.pending_queue = [i for i, t in enumerate(self.tasks) if (t.status or "").strip() == "排队"]
        if not self.pending_queue:
            QtWidgets.QMessageBox.information(self, "提示", "没有需要处理的待处理任务（状态为“排队”）")
            return

        if self.current_worker and self.current_worker.isRunning():
            # 当前正在执行，等待完成后在 _on_finish 中继续
            QtWidgets.QMessageBox.information(self, "提示", "已有任务进行中，已更新队列，完成后将继续执行剩余任务")
            return

        # 立即启动队列中的第一个
        self._start_next_in_queue()

    def _start_next_in_queue(self):
        """启动队列中的下一个“排队”任务。若队列为空则结束。"""
        # 清理无效索引或非排队项
        while self.pending_queue:
            row = self.pending_queue.pop(0)
            if 0 <= row < len(self.tasks) and (self.tasks[row].status or "").strip() == "排队":
                self._start_row(row)
                return
        # 队列耗尽，无需处理
        return

    # 选择控制
    def _select_all_candidates(self):
        for row in range(self.file_table.rowCount()):
            cw = self.file_table.cellWidget(row, 0)
            if not cw:
                continue
            cb = cw.findChild(QtWidgets.QCheckBox, "candCheck")
            if cb:
                cb.setChecked(True)

    def _deselect_all_candidates(self):
        for row in range(self.file_table.rowCount()):
            cw = self.file_table.cellWidget(row, 0)
            if not cw:
                continue
            cb = cw.findChild(QtWidgets.QCheckBox, "candCheck")
            if cb:
                cb.setChecked(False)

    def _schedule_save_config(self):
        """参数变更后延迟保存以减少 IO。"""
        self._save_timer.start()

    def _save_config(self):
        """静默保存当前 GUI 参数到 cache.json（自动保存）。"""
        try:
            # 仅保存已在界面中暴露的字段（其余保持原值）
            self.cfg.text_llm.base_url = self.edit_base_url.text() or None
            self.cfg.text_llm.api_key = self.edit_api_key.text() or None
            self.cfg.text_llm.model = self.edit_model.text()
            self.cfg.text_llm.concurrency = int(self.spin_text_conc.value())
            self.cfg.mm_llm.base_url = self.edit_mm_base_url.text() or None
            self.cfg.mm_llm.api_key = self.edit_mm_api_key.text() or None
            self.cfg.mm_llm.model = self.edit_mm_model.text()
            self.cfg.mm_llm.concurrency = int(self.spin_mm_conc.value())
            # 笔记模式
            data = self.combo_note_mode.currentData()
            mode = data if isinstance(data, str) else 'subtitle'
            # 若缺少 note 字段，补默认
            if not getattr(self.cfg, 'note', None):
                from core.config.schema import NoteConfig
                self.cfg.note = NoteConfig()
            self.cfg.note.mode = mode
            # 保存笔记目录与截图目录
            txt_note_dir = (self.edit_note_dir.text() or '').strip()
            self.cfg.note.note_dir = Path(txt_note_dir) if txt_note_dir else None
            txt_shot_dir = (self.edit_screenshot_dir.text() or '').strip()
            self.cfg.note.screenshot_dir = Path(txt_shot_dir) if txt_shot_dir else None
            ConfigCache().save(self.cfg)
        except Exception:
            # 静默失败，不打断用户操作；关闭时会再尝试
            pass

    def _pick_note_dir(self):
        """选择笔记目录（Markdown 输出根目录）。"""
        try:
            init_dir = self.edit_note_dir.text().strip() or str(Path.cwd())
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择笔记目录", init_dir)
            if d:
                self.edit_note_dir.setText(d)
        except Exception:
            pass

    def _on_table_double_click(self, row: int, column: int):  # noqa: ARG002
        """双击表格行：打开输出目录（笔记目录优先）。"""
        self._open_output_dir()

    def _open_output_dir(self):
        """打开输出目录：优先“笔记目录”，否则 `outputs/` 根目录。"""
        try:
            note_dir = getattr(getattr(self.cfg, 'note', None), 'note_dir', None)
            base = Path(note_dir) if note_dir else Path(self.cfg.export.outputs_root)
            base.mkdir(parents=True, exist_ok=True)
            url = QtCore.QUrl.fromLocalFile(str(base))
            QtGui.QDesktopServices.openUrl(url)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "打开目录失败", str(e))

    def _pick_screenshot_dir(self):
        """选择截图目录（高清重拍输出目录）。"""
        try:
            init_dir = self.edit_screenshot_dir.text().strip() or str(Path.cwd())
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择截图目录", init_dir)
            if d:
                self.edit_screenshot_dir.setText(d)
        except Exception:
            pass

    def _on_progress(self, row: int, p: int):
        self.tasks[row].progress = p
        self._refresh_table()

    def _on_status(self, row: int, s: str):
        self.tasks[row].status = s
        self._refresh_table()

    def _on_finish(self, row: int, status: str, error: str):
        self.tasks[row].status = status
        self.tasks[row].error = error
        self._refresh_table()
        self.current_worker = None
        # 若存在待执行队列，则继续下一个任务
        self._start_next_in_queue()

    def closeEvent(self, event):  # type: ignore[override]
        # 关闭窗口时尝试保存一次配置（静默，失败也不阻塞关闭）
        try:
            self._save_config()
        except Exception:
            pass
        return super().closeEvent(event)

    # 测试连通性（简单结构化返回）
    def _test_text_llm(self):
        try:
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field
            from langchain_core.messages import HumanMessage

            class Ping(BaseModel):
                ok: bool = Field(description="是否连通")

            model = ChatOpenAI(
                api_key=self.edit_api_key.text() or None,
                base_url=self.edit_base_url.text() or None,
                model=self.edit_model.text(),
                temperature=0,
                timeout=30,
            ).with_structured_output(Ping)
            result = model.invoke([HumanMessage(content="返回 {\"ok\": true}")])
            QtWidgets.QMessageBox.information(self, "文本LLM", f"返回：{result.model_dump()}")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "文本LLM错误", str(e))

    def _test_mm_llm(self):
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            # 发送一个极简的纯文本请求（多数多模态端点支持纯文本）
            model = ChatOpenAI(
                api_key=self.edit_mm_api_key.text() or None,
                base_url=self.edit_mm_base_url.text() or None,
                model=self.edit_mm_model.text(),
                temperature=0,
                timeout=30,
            )
            prompt = "请只返回一个数字（1-9），不要任何其他字符、空格或换行。若无法判断返回 5。"
            result = model.invoke([HumanMessage(content=prompt)])
            text = getattr(result, "content", "").strip()
            if not text:
                text = str(result)
            QtWidgets.QMessageBox.information(self, "多模态LLM", f"返回：{text}")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "多模态LLM错误", str(e))

    # 样式与资源
    def _icon(self, name: str) -> QtGui.QIcon:
        """从 core/gui/assets/icons 加载图标。"""
        try:
            base = Path(__file__).resolve().parent / "assets" / "icons"
            p = base / f"{name}.svg"
            if p.exists():
                return QtGui.QIcon(str(p))
        except Exception:
            pass
        return QtGui.QIcon()

    def _apply_styles(self):
        """应用统一 QSS 样式（深色风格，参考同类布局风格）。"""
        qss = """
        /* 全局背景与前景 */
        QWidget { background-color: #202225; color: #E6E6E6; font-size: 13px; }

        /* 分组框 */
        QGroupBox { border: 1px solid #3A3D41; border-radius: 6px; margin-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9AA0A6; }

        /* 标签页 */
        QTabWidget::pane { border: 1px solid #3A3D41; background: #2B2F33; }
        QTabBar::tab { background: #3A3D41; color: #E6E6E6; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
        QTabBar::tab:selected { background: #2B2F33; border-bottom: 2px solid #4EA1FF; }
        QTabBar::tab:hover { background: #454A50; }

        /* 表格 */
        QHeaderView::section { background: #2B2F33; color: #C9D1D9; padding: 6px; border: 0; border-right: 1px solid #3A3D41; }
        QTableWidget { gridline-color: #2F3338; selection-background-color: #2F6FED; selection-color: #FFFFFF; }
        QTableWidget::item:selected { background: #2F6FED; }

        /* 输入控件 */
        QLineEdit, QSpinBox, QComboBox { background: #2B2F33; border: 1px solid #3A3D41; border-radius: 4px; padding: 4px 6px; }
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border: 1px solid #4EA1FF; }

        /* 按钮 */
        QPushButton { background: #3A3D41; border: 1px solid #474B50; border-radius: 6px; padding: 6px 12px; }
        QPushButton:hover { background: #454A50; }
        QPushButton:pressed { background: #2F3338; }

        /* 分割条 */
        QSplitter::handle { background: #2B2F33; }
        QSplitter::handle:hover { background: #3A3D41; }

        /* 滚动条（简化） */
        QScrollBar:vertical, QScrollBar:horizontal { background: #2B2F33; }
        """
        try:
            self.setStyleSheet(qss)
        except Exception:
            pass


def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
