from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from time import perf_counter
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QDesktopServices

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
            exporter = MarkdownExporter(self.cfg.export.outputs_root)
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("note-gen：字幕转分层笔记（含自动截帧）")
        self.resize(1200, 720)

        self.cache = ConfigCache()
        self.cfg = self.cache.load()

        self._build_ui()
        self._connect_signals()

        self.current_worker: Optional[Worker] = None
        self.tasks: list[TaskItem] = []
        # 自动保存防抖计时器
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)  # ms
        self._save_timer.timeout.connect(self._save_config)

    def _build_ui(self):
        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # 左侧：上-文件选择，下-参数配置
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        # 文件选择区
        file_group = QtWidgets.QGroupBox("选择文件/文件夹并添加任务")
        file_layout = QtWidgets.QVBoxLayout(file_group)
        self.btn_pick_video = QtWidgets.QPushButton("选择视频文件")
        self.btn_pick_sub = QtWidgets.QPushButton("选择字幕文件(ASS/VTT/SRT)")
        self.lbl_video = QtWidgets.QLabel("未选择视频")
        self.lbl_sub = QtWidgets.QLabel("未选择字幕")
        self.btn_add_task = QtWidgets.QPushButton("添加到右侧任务列表")
        file_layout.addWidget(self.btn_pick_video)
        file_layout.addWidget(self.lbl_video)
        file_layout.addWidget(self.btn_pick_sub)
        file_layout.addWidget(self.lbl_sub)
        file_layout.addWidget(self.btn_add_task)

        # 参数配置区（仅放关键参数，更多细节可扩展）
        cfg_group = QtWidgets.QGroupBox("参数配置与连通性测试")
        cfg_layout = QtWidgets.QFormLayout(cfg_group)
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

        left_layout.addWidget(file_group)
        left_layout.addWidget(cfg_group)

        # 右侧：任务列表
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["任务ID", "视频", "字幕", "状态", "进度%", "错误"])
        self.btn_start = QtWidgets.QPushButton("开始处理选中任务")
        self.btn_open_output = QtWidgets.QPushButton("打开输出目录")
        right_layout.addWidget(self.table)
        right_layout.addWidget(self.btn_start)
        right_layout.addWidget(self.btn_open_output)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([450, 750])

    def _connect_signals(self):
        self.btn_pick_video.clicked.connect(self._pick_video)
        self.btn_pick_sub.clicked.connect(self._pick_sub)
        self.btn_add_task.clicked.connect(self._add_task)
        self.btn_start.clicked.connect(self._start_selected)
        self.btn_open_output.clicked.connect(self._open_outputs)
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

    # 左侧交互
    def _pick_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.mkv *.mov *.avi)")
        if path:
            self.lbl_video.setText(path)

    def _pick_sub(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择字幕", "", "字幕文件 (*.srt *.ass *.ssa *.vtt)")
        if path:
            self.lbl_sub.setText(path)

    def _add_task(self):
        v = self.lbl_video.text()
        s = self.lbl_sub.text()
        if not v or v == "未选择视频" or not s or s == "未选择字幕":
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择视频与字幕")
            return
        task = TaskItem(video=Path(v), subtitle=Path(s))
        self.tasks.append(task)
        self._refresh_table()

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

    def _open_outputs(self):
        Path(self.cfg.export.outputs_root).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.cfg.export.outputs_root)))

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


def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
