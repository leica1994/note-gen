from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from time import perf_counter
from PySide6 import QtCore, QtWidgets, QtGui

from core.config.cache import ConfigCache
from core.config.schema import AppConfig
from core.export.markdown import MarkdownExporter
from core.note.generator import NoteGenerator
from core.note.models import GenerationInputMeta
from core.subtitles.loader import load_subtitle
from core.utils.evidence import EvidenceWriter
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

            evidence = EvidenceWriter(self.cfg.export.evidence_root).for_task(self.task.task_id)
            out_dir = Path(self.cfg.export.outputs_root) / self.task.task_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # 初始化任务日志
            logger = init_task_logger(self.task.task_id or "unknown", self.cfg.export.logs_root)
            logger.info("任务开始", extra={
                "video": str(self.task.video),
                "subtitle": str(self.task.subtitle),
            })
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
            generator = NoteGenerator(self.cfg, evidence, logger=logger)
            meta = GenerationInputMeta(video_path=self.task.video, subtitle=sub_doc, params=params)
            t1 = perf_counter()
            note = generator.generate(meta, out_dir)
            logger.info("笔记生成完成", extra={
                "chapters": len(note.chapters),
                "cost_ms": int((perf_counter() - t1) * 1000),
            })
            self.progress_changed.emit(85)
            # 3) 导出 Markdown
            exporter = MarkdownExporter(self.cfg.export.outputs_root, note_mode=(
                self.cfg.note.mode if getattr(self.cfg, 'note', None) else 'subtitle'))
            md = exporter.export(note, filename=f"{self.task.task_id}.md")
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

        # 候选选择控制区：全选 / 全不选 / 添加到任务
        cand_btn_row = QtWidgets.QHBoxLayout()
        self.btn_select_all_cand = QtWidgets.QPushButton("全选")
        self.btn_deselect_all_cand = QtWidgets.QPushButton("全不选")
        self.btn_add_task = QtWidgets.QPushButton("添加到任务列表")
        cand_btn_row.addWidget(self.btn_select_all_cand)
        cand_btn_row.addWidget(self.btn_deselect_all_cand)
        cand_btn_row.addStretch(1)
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
        self.spin_text_conc.setRange(1, 64)
        self.spin_text_conc.setValue(self.cfg.text_llm.concurrency)
        self.btn_test_llm = QtWidgets.QPushButton("测试文本LLM连通性")

        self.edit_mm_base_url = QtWidgets.QLineEdit(self.cfg.mm_llm.base_url or "")
        self.edit_mm_api_key = QtWidgets.QLineEdit(self.cfg.mm_llm.api_key or "")
        self.edit_mm_api_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.edit_mm_model = QtWidgets.QLineEdit(self.cfg.mm_llm.model)
        self.spin_mm_conc = QtWidgets.QSpinBox()
        self.spin_mm_conc.setRange(1, 64)
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
        bottom_tabs.addTab(tab_note, "笔记参数配置")

        # 左侧组装
        left_splitter.addWidget(top_left)
        left_splitter.addWidget(bottom_tabs)
        left_splitter.setSizes([450, 270])

        # ============ 右侧：任务列表 ============
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["任务ID", "视频", "字幕", "状态", "进度%", "错误"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.btn_start = QtWidgets.QPushButton("开始处理选中任务")
        right_layout.addWidget(self.table)
        right_layout.addWidget(self.btn_start)

        # 顶层装配
        root_splitter.addWidget(left_splitter)
        root_splitter.addWidget(right)
        root_splitter.setSizes([520, 680])

    def _connect_signals(self):
        # 左侧：模式切换与选择输入
        self.radio_mode_file.toggled.connect(self._update_pick_button_text)
        self.radio_mode_folder.toggled.connect(self._update_pick_button_text)
        self.btn_pick_input.clicked.connect(self._pick_input)
        self.btn_add_task.clicked.connect(self._add_task)
        self.btn_select_all_cand.clicked.connect(self._select_all_candidates)
        self.btn_deselect_all_cand.clicked.connect(self._deselect_all_candidates)
        self.btn_start.clicked.connect(self._start_selected)
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
        self._refresh_table()

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
            self.table.setItem(i, 5, cell(t.error or ""))
        # 按钮图标（样式增强，不影响逻辑）
        try:
            self.btn_start.setIcon(self._icon("play"))
            self.btn_test_llm.setIcon(self._icon("robot"))
            self.btn_test_mm.setIcon(self._icon("camera"))
        except Exception:
            pass

    def _start_selected(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.tasks):
            return
        if self.current_worker and self.current_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "提示", "已有任务进行中，请稍后")
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
            ConfigCache().save(self.cfg)
        except Exception:
            # 静默失败，不打断用户操作；关闭时会再尝试
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
            from pydantic import BaseModel, Field
            from langchain_core.messages import HumanMessage

            class Pick(BaseModel):
                index: int = Field(ge=1, le=1)

            # 发送一个极简的“无图”请求，仅测试结构化解析（部分多模态端点也支持纯文本）
            model = ChatOpenAI(
                api_key=self.edit_mm_api_key.text() or None,
                base_url=self.edit_mm_base_url.text() or None,
                model=self.edit_mm_model.text(),
                temperature=0,
                timeout=30,
            ).with_structured_output(Pick)
            result = model.invoke([HumanMessage(content="返回 {\"index\": 1}")])
            QtWidgets.QMessageBox.information(self, "多模态LLM", f"返回：{result.model_dump()}")
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
