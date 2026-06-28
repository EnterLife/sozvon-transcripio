from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices
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

from audio.audio_router import AudioRouter, AudioRouterStats
from audio.devices import (
    AudioDiagnostics,
    diagnose_audio,
    list_audio_devices,
    resolve_loopback_index,
    resolve_microphone_index,
)
from audio.loopback_capture import LoopbackCapture
from audio.microphone_capture import MicrophoneCapture
from audio.types import AudioSource
from config.settings import AppSettings, load_settings, save_settings
from core.atomic_write import write_text_atomic
from core.paths import AppPaths
from gui.settings_dialog import SettingsDialog
from gui.status import parse_capture_status
from speech.mock_engine import MockTranscriptionEngine
from speech.model_manager import CalibrationReport, ModelManager
from speech.transcription_worker import TranscriptionWorker
from speech.types import TranscriptEvent
from speech.whisper_engine import WhisperEngine
from storage.session_paths import resolve_transcript_dir, same_transcript_dir
from storage.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)


def capture_sources_for_diagnostics(diagnostics: AudioDiagnostics) -> list[AudioSource]:
    if not diagnostics.sounddevice_available:
        return []
    sources: list[AudioSource] = []
    if diagnostics.microphone_available:
        sources.append(AudioSource.USER_MIC)
    if diagnostics.loopback_available:
        sources.append(AudioSource.REMOTE_AUDIO)
    return sources


def audio_diagnostics_notice(diagnostics: AudioDiagnostics) -> str | None:
    return "\n".join(diagnostics.messages) or None


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
                self.settings.recognition.device,
                self.settings.recognition.compute_type,
                self.settings.recognition.local_model_path,
            )
            self.status.emit(f"Loading {selection.display_name} on {selection.device}")
            model = self.manager.ensure_model(
                selection,
                self.status.emit,
                self.settings.recognition.hf_token,
                self.settings.recognition.auto_install_cuda_runtime,
                self.settings.recognition.device == "auto",
                self.settings.recognition.offline_mode,
            )
            self.loaded.emit(model, selection)
        except Exception as exc:
            logger.exception("Model loading failed")
            self.failed.emit(str(exc))


class CalibrationLoader(QObject):
    completed = Signal(object)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, manager: ModelManager, settings: AppSettings) -> None:
        super().__init__()
        self.manager = manager
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            report = self.manager.calibrate(
                self.settings.recognition.model_size,
                self.settings.recognition.auto_select_model,
                self.settings.recognition.device,
                self.settings.recognition.compute_type,
                self.settings.recognition.local_model_path,
                self.status.emit,
                self.settings.recognition.hf_token,
                self.settings.recognition.auto_install_cuda_runtime,
                self.settings.recognition.offline_mode,
            )
            self.completed.emit(report)
        except Exception as exc:
            logger.exception("Model calibration failed")
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
        self.store = TranscriptStore(
            resolve_transcript_dir(self.settings.storage.transcript_dir, self.paths.transcripts_dir)
        )
        self._last_drop_warning_count = 0

        self.model_label = QLabel("Model: not loaded")
        self.gpu_label = QLabel("GPU: detecting")
        self.mic_label = QLabel("Mic: idle")
        self.loopback_label = QLabel("System audio: idle")
        self.save_path_label = QLabel()
        self.save_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.status_console = QPlainTextEdit()
        self.status_console.setReadOnly(True)
        self.status_console.setMaximumHeight(120)
        self.status_console.setPlaceholderText("Status")
        self.status_console.document().setMaximumBlockCount(200)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.settings_button = QPushButton("Settings")
        self.export_button = QPushButton("Export")
        self.new_session_button = QPushButton("New session")
        self.clear_button = QPushButton("Clear")
        self.open_folder_button = QPushButton("Open folder")
        self.calibrate_button = QPushButton("Calibrate")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        self._build_ui()
        self._connect_signals()
        self._refresh_session_label()
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
        controls.addWidget(self.open_folder_button)
        controls.addWidget(self.calibrate_button)
        controls.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(indicators)
        layout.addWidget(self.save_path_label)
        layout.addWidget(self.transcript, stretch=1)
        layout.addWidget(self.status_console)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())
        self._set_status("Preparing model...")

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.settings_button.clicked.connect(self._open_settings)
        self.export_button.clicked.connect(self._export_transcript)
        self.new_session_button.clicked.connect(self._new_session)
        self.clear_button.clicked.connect(self._clear_session)
        self.open_folder_button.clicked.connect(self._open_transcript_folder)
        self.calibrate_button.clicked.connect(self._calibrate_model)
        self.transcript_event.connect(self._on_transcript_event)
        self.worker_error.connect(self._show_error)
        self.capture_status.connect(self._on_capture_status)

    def _apply_default_devices(self) -> None:
        if self.settings.audio.microphone_device is not None:
            self.settings.audio.microphone_device = resolve_microphone_index(
                self.settings.audio.microphone_device
            )
        if self.settings.audio.loopback_device is not None:
            self.settings.audio.loopback_device = resolve_loopback_index(
                self.settings.audio.loopback_device
            )

    def _prepare_recognition(self) -> None:
        if self.settings.recognition.dry_run:
            self.model = object()
            self.model_selection = None
            self.model_label.setText("Model: test mode")
            self.gpu_label.setText("Acceleration: test")
            self._set_status("Ready in test mode")
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
        self.model_loader.status.connect(self._set_status)
        self.model_loader.loaded.connect(self.model_thread.quit)
        self.model_loader.failed.connect(self.model_thread.quit)
        self.model_thread.start()

    @Slot(object, object)
    def _on_model_loaded(self, model, selection) -> None:
        self.model = model
        self.model_selection = selection
        if selection.is_local_model:
            self.settings.recognition.local_model_path = str(selection.local_model_path)
        else:
            self.settings.recognition.model_size = selection.model_size
        save_settings(self.paths.settings_file, self.settings)
        self.model_label.setText(f"Model: {selection.display_name}")
        gpu_state = "CUDA" if selection.hardware.cuda_available else "CPU"
        self.gpu_label.setText(f"Acceleration: {gpu_state}")
        self._set_status("Ready")
        self.start_button.setEnabled(True)

    @Slot(str)
    def _on_model_failed(self, message: str) -> None:
        self.model_label.setText("Model: failed")
        self._set_status("Model loading failed")
        self._show_error(message)

    def _start(self) -> None:
        if self.model is None:
            self._show_error("Model is not loaded yet.")
            return

        diagnostics = diagnose_audio(
            self.settings.audio.microphone_device,
            self.settings.audio.loopback_device,
        )
        if not diagnostics.sounddevice_available:
            self._show_error("\n".join(diagnostics.messages) or "Audio input is not available.")
            return
        capture_sources = capture_sources_for_diagnostics(diagnostics)
        if not capture_sources:
            self._show_error("\n".join(diagnostics.messages) or "No usable audio source is available.")
            return
        notice = audio_diagnostics_notice(diagnostics)
        if notice:
            self._append_status("WARN", notice)

        self._last_drop_warning_count = 0
        self.router = AudioRouter(on_backpressure=self._on_audio_backpressure)
        sample_rate = self.settings.audio.sample_rate
        chunk_seconds = self.settings.audio.chunk_seconds
        self.captures = []
        if AudioSource.USER_MIC in capture_sources:
            self.captures.append(
                MicrophoneCapture(
                    self.settings.audio.microphone_device,
                    sample_rate,
                    chunk_seconds,
                    self.router.push,
                    self.worker_error.emit,
                    self.capture_status.emit,
                )
            )
        if AudioSource.REMOTE_AUDIO in capture_sources:
            self.captures.append(
                LoopbackCapture(
                    self.settings.audio.loopback_device,
                    sample_rate,
                    chunk_seconds,
                    self.router.push,
                    self.worker_error.emit,
                    self.capture_status.emit,
                )
            )
        self.mic_label.setText(
            "Mic: starting" if AudioSource.USER_MIC in capture_sources else "Mic: unavailable"
        )
        self.loopback_label.setText(
            "System audio: starting"
            if AudioSource.REMOTE_AUDIO in capture_sources
            else "System audio: unavailable"
        )
        engine = (
            MockTranscriptionEngine()
            if self.settings.recognition.dry_run
            else WhisperEngine(
                self.model,
                self.settings.recognition.language,
                self.settings.recognition.quality_mode,
                self.settings.recognition.glossary_terms,
                self.settings.recognition.word_timestamps,
            )
        )
        self.transcriber = TranscriptionWorker(
            self.router,
            engine,
            self.transcript_event.emit,
            self.worker_error.emit,
            self.settings.recognition.transcription_window_seconds,
        )

        for capture in self.captures:
            capture.start()
        self.transcriber.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_status("Transcribing...")
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
        self._set_status("Stopped")
        logger.info("Transcription stopped")

    def _open_settings(self) -> None:
        previous_microphone = self.settings.audio.microphone_device
        previous_loopback = self.settings.audio.loopback_device
        previous_sample_rate = self.settings.audio.sample_rate
        previous_chunk_seconds = self.settings.audio.chunk_seconds
        previous_dry_run = self.settings.recognition.dry_run
        previous_model = self.settings.recognition.model_size
        previous_local_model_path = self.settings.recognition.local_model_path
        previous_quality_mode = self.settings.recognition.quality_mode
        previous_glossary_terms = self.settings.recognition.glossary_terms
        previous_word_timestamps = self.settings.recognition.word_timestamps
        previous_auto = self.settings.recognition.auto_select_model
        previous_device = self.settings.recognition.device
        previous_compute_type = self.settings.recognition.compute_type
        previous_transcription_window = self.settings.recognition.transcription_window_seconds
        previous_cuda_runtime = self.settings.recognition.auto_install_cuda_runtime
        previous_hf_token = self.settings.recognition.hf_token
        previous_offline_mode = self.settings.recognition.offline_mode
        previous_transcript_dir = self.store.directory
        dialog = SettingsDialog(self.settings, list_audio_devices(), self)
        if dialog.exec():
            save_settings(self.paths.settings_file, self.settings)
            self.autosave.setInterval(max(5, self.settings.storage.autosave_seconds) * 1000)
            selected_transcript_dir = resolve_transcript_dir(
                self.settings.storage.transcript_dir,
                self.paths.transcripts_dir,
            )
            if not same_transcript_dir(previous_transcript_dir, selected_transcript_dir):
                self._save_transcript()
                self.store = TranscriptStore(selected_transcript_dir)
                self.transcript.clear()
                self._refresh_session_label()
                self._set_status("Transcript folder changed; new session started")
            else:
                self._refresh_session_label()
                self._set_status("Settings saved")
            recognition_changed = (
                previous_dry_run != self.settings.recognition.dry_run
                or previous_model != self.settings.recognition.model_size
                or previous_local_model_path != self.settings.recognition.local_model_path
                or previous_quality_mode != self.settings.recognition.quality_mode
                or previous_glossary_terms != self.settings.recognition.glossary_terms
                or previous_word_timestamps != self.settings.recognition.word_timestamps
                or previous_auto != self.settings.recognition.auto_select_model
                or previous_device != self.settings.recognition.device
                or previous_compute_type != self.settings.recognition.compute_type
                or previous_cuda_runtime != self.settings.recognition.auto_install_cuda_runtime
                or previous_hf_token != self.settings.recognition.hf_token
                or previous_offline_mode != self.settings.recognition.offline_mode
            )
            capture_settings_changed = (
                previous_microphone != self.settings.audio.microphone_device
                or previous_loopback != self.settings.audio.loopback_device
                or previous_sample_rate != self.settings.audio.sample_rate
                or previous_chunk_seconds != self.settings.audio.chunk_seconds
                or previous_transcription_window
                != self.settings.recognition.transcription_window_seconds
            )
            if recognition_changed:
                self._stop()
                self._prepare_recognition()
            elif capture_settings_changed and self.transcriber is not None:
                self._stop()
                self._set_status("Recording stopped to apply audio settings")
            elif self.model is None:
                self._prepare_recognition()

    @Slot(object)
    def _on_transcript_event(self, event: TranscriptEvent) -> None:
        record = self.store.add(event)
        self.transcript.appendPlainText(f"[{record.timestamp}] {record.speaker}:\n{record.text}\n")

    @Slot(str)
    def _show_error(self, message: str) -> None:
        logger.error(message)
        self._mark_source_failed(message)
        self._append_status("ERROR", message)
        QMessageBox.warning(self, "Realtime Call Transcriber", message)

    @Slot(str)
    def _on_capture_status(self, message: str) -> None:
        self._set_status(message)
        source, state = parse_capture_status(message)
        if source is AudioSource.USER_MIC:
            self.mic_label.setText(f"Mic: {state}")
        elif source is AudioSource.REMOTE_AUDIO:
            self.loopback_label.setText(f"System audio: {state}")

    def _on_audio_backpressure(self, stats: AudioRouterStats) -> None:
        if stats.dropped_chunks == 1 or stats.dropped_chunks >= self._last_drop_warning_count + 10:
            self._last_drop_warning_count = stats.dropped_chunks
            self.capture_status.emit(
                f"Transcription is behind; dropped {stats.dropped_chunks} audio chunks"
            )

    @Slot(str)
    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self._append_status("INFO", message)

    def _mark_source_failed(self, message: str) -> None:
        lowered = message.lower()
        if "loopback" in lowered or "wasapi" in lowered or "remote_audio" in lowered:
            self.loopback_label.setText("System audio: failed")
        if "microphone" in lowered or "user_mic" in lowered:
            self.mic_label.setText("Mic: failed")

    def _append_status(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_console.appendPlainText(f"[{timestamp}] {level}: {message}")
        scrollbar = self.status_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _save_transcript(self) -> None:
        try:
            self.store.save()
        except OSError as exc:
            logger.exception("Could not save transcript")
            self.statusBar().showMessage("Could not save transcript")
            self._append_status("ERROR", f"Could not save transcript: {exc}")

    def _export_transcript(self) -> None:
        self._save_transcript()
        target, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export transcript",
            str(self.store.txt_path),
            "Text files (*.txt);;Markdown files (*.md);;JSON files (*.json)",
        )
        if not target:
            return
        target_path = Path(target)
        if target_path.suffix.lower() == ".json":
            source = self.store.to_json()
        elif target_path.suffix.lower() == ".md":
            source = self.store.to_markdown()
        else:
            source = self.store.to_text()
        write_text_atomic(target_path, source)
        self._set_status(f"Exported to {target_path}")

    def _new_session(self) -> None:
        self._save_transcript()
        self.store = TranscriptStore(
            resolve_transcript_dir(self.settings.storage.transcript_dir, self.paths.transcripts_dir)
        )
        self.transcript.clear()
        self._refresh_session_label()
        self._set_status("New transcript session started")

    def _clear_session(self) -> None:
        self.store.clear()
        self.transcript.clear()
        self._save_transcript()
        self._set_status("Transcript cleared")

    def _open_transcript_folder(self) -> None:
        self.store.directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.directory)))

    def _refresh_session_label(self) -> None:
        self.save_path_label.setText(f"Autosave: {self.store.txt_path}")

    def _calibrate_model(self) -> None:
        if self.settings.recognition.dry_run:
            self._show_error("Calibration is unavailable in test mode.")
            return
        self._stop()
        self.start_button.setEnabled(False)
        self.calibrate_button.setEnabled(False)
        self.model_label.setText("Model: calibrating")
        self.calibration_thread = QThread(self)
        self.calibration_loader = CalibrationLoader(ModelManager(self.paths.models_dir), self.settings)
        self.calibration_loader.moveToThread(self.calibration_thread)
        self.calibration_thread.started.connect(self.calibration_loader.run)
        self.calibration_loader.completed.connect(self._on_calibration_completed)
        self.calibration_loader.failed.connect(self._on_calibration_failed)
        self.calibration_loader.status.connect(self._set_status)
        self.calibration_loader.completed.connect(self.calibration_thread.quit)
        self.calibration_loader.failed.connect(self.calibration_thread.quit)
        self.calibration_thread.finished.connect(lambda: self.calibrate_button.setEnabled(True))
        self.calibration_thread.start()

    @Slot(object)
    def _on_calibration_completed(self, report: CalibrationReport) -> None:
        selected = report.selected
        if selected is None:
            self.model_label.setText("Model: calibration failed")
            self._show_error("Calibration did not find a usable model.")
            return
        result_lines = [
            f"{result.model_size}: RTF {result.realtime_factor:.2f}"
            for result in report.results
            if result.error is None
        ]
        if not selected.passed:
            self.model_label.setText(f"Model: {selected.model_size}")
            self.start_button.setEnabled(self.model is not None)
            self._set_status(
                f"No calibrated model met real-time target; best RTF {selected.realtime_factor:.2f}"
            )
            if result_lines:
                self._append_status("INFO", "Calibration results: " + "; ".join(result_lines))
            return
        if self.settings.recognition.local_model_path is None:
            self.settings.recognition.model_size = selected.model_size
            self.settings.recognition.auto_select_model = False
            save_settings(self.paths.settings_file, self.settings)
            self._set_status(f"Calibration selected {selected.model_size}")
            self._prepare_recognition()
        else:
            self._set_status(f"Local model RTF {selected.realtime_factor:.2f}")
            self._prepare_recognition()
        if result_lines:
            self._append_status("INFO", "Calibration results: " + "; ".join(result_lines))

    @Slot(str)
    def _on_calibration_failed(self, message: str) -> None:
        self.model_label.setText("Model: calibration failed")
        self.start_button.setEnabled(self.model is not None)
        self._show_error(message)
