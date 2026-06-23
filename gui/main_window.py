from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from audio.audio_router import AudioRouter
from audio.devices import (
    default_loopback_index,
    default_microphone_index,
    diagnose_audio,
    list_audio_devices,
)
from audio.loopback_capture import LoopbackCapture
from audio.microphone_capture import MicrophoneCapture
from config.settings import AppSettings, load_settings, save_settings
from core.atomic_write import write_text_atomic
from core.paths import AppPaths
from gui.settings_dialog import SettingsDialog
from speech.mock_engine import MockTranscriptionEngine
from speech.model_manager import ModelManager
from speech.transcription_worker import TranscriptionWorker
from speech.types import TranscriptEvent
from speech.whisper_engine import WhisperEngine
from storage.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)


class ModelLoader(QObject):
    loaded = Signal(object, object)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, manager: ModelManager, settings: AppSettings) -> None:
        super().__init__()
        self.manager = manager
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            selection = self.manager.select(
                self.settings.recognition.model_size,
                self.settings.recognition.auto_select_model,
            )
            self.status.emit(f"Loading {selection.model_size} on {selection.device}")
            model = self.manager.ensure_model(selection, self.status.emit)
            self.loaded.emit(model, selection)
        except Exception as exc:
            logger.exception("Model loading failed")
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    transcript_event = Signal(object)
    worker_error = Signal(str)
    capture_status = Signal(str)

    def __init__(self, paths: AppPaths) -> None:
        super().__init__()
        self.paths = paths
        self.settings = load_settings(paths.settings_file)
        if self.settings.storage.transcript_dir is None:
            self.settings.storage.transcript_dir = str(paths.transcripts_dir)
        self._apply_default_devices()
        save_settings(paths.settings_file, self.settings)

        self.router: AudioRouter | None = None
        self.captures = []
        self.transcriber: TranscriptionWorker | None = None
        self.model = None
        self.model_selection = None
        self.store = TranscriptStore(Path(self.settings.storage.transcript_dir))

        self.model_label = QLabel("Model: not loaded")
        self.gpu_label = QLabel("GPU: detecting")
        self.mic_label = QLabel("Mic: idle")
        self.loopback_label = QLabel("System audio: idle")
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.settings_button = QPushButton("Settings")
        self.export_button = QPushButton("Export")
        self.new_session_button = QPushButton("New session")
        self.clear_button = QPushButton("Clear")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        self._build_ui()
        self._connect_signals()
        self._prepare_recognition()

        self.autosave = QTimer(self)
        self.autosave.setInterval(max(5, self.settings.storage.autosave_seconds) * 1000)
        self.autosave.timeout.connect(self._save_transcript)
        self.autosave.start()

    def closeEvent(self, event) -> None:
        self._stop()
        self._save_transcript()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self.setWindowTitle("Realtime Call Transcriber")

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._open_settings)
        toolbar.addAction(settings_action)
        self.addToolBar(toolbar)

        indicators = QHBoxLayout()
        for label in (self.mic_label, self.loopback_label, self.model_label, self.gpu_label):
            label.setMinimumWidth(150)
            indicators.addWidget(label)
        indicators.addStretch(1)

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.settings_button)
        controls.addWidget(self.export_button)
        controls.addWidget(self.new_session_button)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(indicators)
        layout.addWidget(self.transcript, stretch=1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Preparing model...")

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.settings_button.clicked.connect(self._open_settings)
        self.export_button.clicked.connect(self._export_transcript)
        self.new_session_button.clicked.connect(self._new_session)
        self.clear_button.clicked.connect(self._clear_session)
        self.transcript_event.connect(self._on_transcript_event)
        self.worker_error.connect(self._show_error)
        self.capture_status.connect(self.statusBar().showMessage)

    def _apply_default_devices(self) -> None:
        if self.settings.audio.microphone_device is None:
            self.settings.audio.microphone_device = default_microphone_index()
        if self.settings.audio.loopback_device is None:
            self.settings.audio.loopback_device = default_loopback_index()

    def _prepare_recognition(self) -> None:
        if self.settings.recognition.dry_run:
            self.model = object()
            self.model_selection = None
            self.model_label.setText("Model: test mode")
            self.gpu_label.setText("Acceleration: test")
            self.statusBar().showMessage("Ready in test mode")
            self.start_button.setEnabled(True)
            return
        self._load_model_async()

    def _load_model_async(self) -> None:
        self.start_button.setEnabled(False)
        self.model = None
        self.model_label.setText("Model: loading")
        self.model_thread = QThread(self)
        self.model_loader = ModelLoader(ModelManager(self.paths.models_dir), self.settings)
        self.model_loader.moveToThread(self.model_thread)
        self.model_thread.started.connect(self.model_loader.run)
        self.model_loader.loaded.connect(self._on_model_loaded)
        self.model_loader.failed.connect(self._on_model_failed)
        self.model_loader.status.connect(self.statusBar().showMessage)
        self.model_loader.loaded.connect(self.model_thread.quit)
        self.model_loader.failed.connect(self.model_thread.quit)
        self.model_thread.start()

    @Slot(object, object)
    def _on_model_loaded(self, model, selection) -> None:
        self.model = model
        self.model_selection = selection
        self.settings.recognition.model_size = selection.model_size
        save_settings(self.paths.settings_file, self.settings)
        self.model_label.setText(f"Model: {selection.model_size}")
        gpu_state = "CUDA" if selection.hardware.cuda_available else "CPU"
        self.gpu_label.setText(f"Acceleration: {gpu_state}")
        self.statusBar().showMessage("Ready")
        self.start_button.setEnabled(True)

    @Slot(str)
    def _on_model_failed(self, message: str) -> None:
        self.model_label.setText("Model: failed")
        self.statusBar().showMessage("Model loading failed")
        self._show_error(message)

    def _start(self) -> None:
        if self.model is None:
            self._show_error("Model is not loaded yet.")
            return

        diagnostics = diagnose_audio(
            self.settings.audio.microphone_device,
            self.settings.audio.loopback_device,
        )
        if not diagnostics.sounddevice_available or not diagnostics.microphone_available:
            self._show_error("\n".join(diagnostics.messages) or "Audio input is not available.")
            return
        if diagnostics.messages:
            QMessageBox.information(self, "Audio diagnostics", "\n".join(diagnostics.messages))

        self.router = AudioRouter()
        sample_rate = self.settings.audio.sample_rate
        chunk_seconds = self.settings.audio.chunk_seconds
        self.captures = [
            MicrophoneCapture(
                self.settings.audio.microphone_device,
                sample_rate,
                chunk_seconds,
                self.router.push,
                self.worker_error.emit,
                self.capture_status.emit,
            ),
            LoopbackCapture(
                self.settings.audio.loopback_device,
                sample_rate,
                chunk_seconds,
                self.router.push,
                self.worker_error.emit,
                self.capture_status.emit,
            ),
        ]
        engine = (
            MockTranscriptionEngine()
            if self.settings.recognition.dry_run
            else WhisperEngine(self.model, self.settings.recognition.language)
        )
        self.transcriber = TranscriptionWorker(
            self.router,
            engine,
            self.transcript_event.emit,
            self.worker_error.emit,
        )

        for capture in self.captures:
            capture.start()
        self.transcriber.start()
        self.mic_label.setText("Mic: active")
        self.loopback_label.setText("System audio: active")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("Transcribing...")
        logger.info("Transcription started")

    def _stop(self) -> None:
        for capture in self.captures:
            capture.stop()
        self.captures = []
        if self.transcriber:
            self.transcriber.stop()
            self.transcriber = None
        self.mic_label.setText("Mic: idle")
        self.loopback_label.setText("System audio: idle")
        self.start_button.setEnabled(self.model is not None)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Stopped")
        logger.info("Transcription stopped")

    def _open_settings(self) -> None:
        previous_dry_run = self.settings.recognition.dry_run
        previous_model = self.settings.recognition.model_size
        previous_auto = self.settings.recognition.auto_select_model
        dialog = SettingsDialog(self.settings, list_audio_devices(), self)
        if dialog.exec():
            save_settings(self.paths.settings_file, self.settings)
            self.autosave.setInterval(max(5, self.settings.storage.autosave_seconds) * 1000)
            self.store = TranscriptStore(Path(self.settings.storage.transcript_dir or self.paths.transcripts_dir))
            self.statusBar().showMessage("Settings saved")
            recognition_changed = (
                previous_dry_run != self.settings.recognition.dry_run
                or previous_model != self.settings.recognition.model_size
                or previous_auto != self.settings.recognition.auto_select_model
            )
            if recognition_changed:
                self._stop()
                self._prepare_recognition()
            elif self.model is None:
                self._prepare_recognition()

    @Slot(object)
    def _on_transcript_event(self, event: TranscriptEvent) -> None:
        record = self.store.add(event)
        self.transcript.appendPlainText(f"[{record.timestamp}] {record.speaker}:\n{record.text}\n")

    @Slot(str)
    def _show_error(self, message: str) -> None:
        logger.error(message)
        QMessageBox.warning(self, "Realtime Call Transcriber", message)

    def _save_transcript(self) -> None:
        self.store.save()

    def _export_transcript(self) -> None:
        self._save_transcript()
        target, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export transcript",
            str(self.store.txt_path),
            "Text files (*.txt);;JSON files (*.json)",
        )
        if not target:
            return
        target_path = Path(target)
        source = self.store.json_path if target_path.suffix.lower() == ".json" else self.store.txt_path
        write_text_atomic(target_path, source.read_text(encoding="utf-8"))
        self.statusBar().showMessage(f"Exported to {target_path}")

    def _new_session(self) -> None:
        self._save_transcript()
        self.store = TranscriptStore(Path(self.settings.storage.transcript_dir or self.paths.transcripts_dir))
        self.transcript.clear()
        self.statusBar().showMessage("New transcript session started")

    def _clear_session(self) -> None:
        self.store.clear()
        self.transcript.clear()
        self._save_transcript()
        self.statusBar().showMessage("Transcript cleared")
