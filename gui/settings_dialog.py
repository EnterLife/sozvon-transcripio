from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QDoubleSpinBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from audio.devices import AudioDevice
from config.settings import AppSettings


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, devices: list[AudioDevice], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings
        self.devices = devices

        self.mic_combo = QComboBox()
        self.loopback_combo = QComboBox()
        self.language_combo = QComboBox()
        self.model_combo = QComboBox()
        self.auto_model = QCheckBox("Auto-select")
        self.hf_token = QLineEdit()
        self.dry_run = QCheckBox("Use test transcript engine")
        self.sample_rate = QComboBox()
        self.chunk_seconds = QDoubleSpinBox()
        self.autosave_seconds = QSpinBox()
        self.transcript_dir = QLineEdit()
        self.browse_button = QPushButton("Browse")

        self._populate()

        form = QFormLayout()
        form.addRow("Microphone", self.mic_combo)
        form.addRow("Output / loopback", self.loopback_combo)
        form.addRow("Language", self.language_combo)
        form.addRow("Model", self.model_combo)
        form.addRow("Model selection", self.auto_model)
        form.addRow("Hugging Face token", self.hf_token)
        form.addRow("Test mode", self.dry_run)
        form.addRow("Sample rate", self.sample_rate)
        form.addRow("Chunk seconds", self.chunk_seconds)
        form.addRow("Autosave seconds", self.autosave_seconds)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.transcript_dir)
        path_layout.addWidget(self.browse_button)
        form.addRow("Transcript folder", path_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.browse_button.clicked.connect(self._browse_transcript_dir)
        self.auto_model.toggled.connect(self._update_model_enabled)

    def accept(self) -> None:
        self.settings.audio.microphone_device = self.mic_combo.currentData()
        self.settings.audio.loopback_device = self.loopback_combo.currentData()
        self.settings.audio.sample_rate = self.sample_rate.currentData()
        self.settings.audio.chunk_seconds = self.chunk_seconds.value()
        self.settings.recognition.language = self.language_combo.currentData()
        self.settings.recognition.model_size = self.model_combo.currentData()
        self.settings.recognition.auto_select_model = self.auto_model.isChecked()
        self.settings.recognition.hf_token = self.hf_token.text().strip() or None
        self.settings.recognition.dry_run = self.dry_run.isChecked()
        self.settings.storage.autosave_seconds = self.autosave_seconds.value()
        self.settings.storage.transcript_dir = self.transcript_dir.text().strip() or None
        super().accept()

    def _populate(self) -> None:
        self._add_device_options(self.mic_combo, input_only=True)
        self._add_device_options(self.loopback_combo, input_only=False)

        for language, label in (("ru", "Russian"), ("en", "English"), ("", "Auto")):
            self.language_combo.addItem(label, language)
        self._select_data(self.language_combo, self.settings.recognition.language)

        for model in ("tiny", "base", "small", "medium", "large-v3"):
            self.model_combo.addItem(model, model)
        self._select_data(self.model_combo, self.settings.recognition.model_size or "base")
        self.auto_model.setChecked(self.settings.recognition.auto_select_model)
        self.hf_token.setEchoMode(QLineEdit.Password)
        self.hf_token.setPlaceholderText("Optional token for model downloads")
        self.hf_token.setText(self.settings.recognition.hf_token or "")
        self.dry_run.setChecked(self.settings.recognition.dry_run)
        self._update_model_enabled()

        for sample_rate in (8000, 16000, 22050, 44100, 48000):
            self.sample_rate.addItem(f"{sample_rate} Hz", sample_rate)
        self._select_data(self.sample_rate, self.settings.audio.sample_rate)

        self.chunk_seconds.setRange(0.5, 2.0)
        self.chunk_seconds.setSingleStep(0.25)
        self.chunk_seconds.setDecimals(2)
        self.chunk_seconds.setValue(self.settings.audio.chunk_seconds)

        self.autosave_seconds.setRange(5, 3600)
        self.autosave_seconds.setValue(self.settings.storage.autosave_seconds)
        self.transcript_dir.setText(self.settings.storage.transcript_dir or "")

    def _add_device_options(self, combo: QComboBox, input_only: bool) -> None:
        combo.addItem("Default", None)
        for device in self.devices:
            if input_only and device.max_input_channels <= 0:
                continue
            if not input_only and device.max_output_channels <= 0:
                continue
            combo.addItem(f"{device.name} ({device.hostapi})", device.index)

        selected = (
            self.settings.audio.microphone_device if input_only else self.settings.audio.loopback_device
        )
        self._select_data(combo, selected)

    def _select_data(self, combo: QComboBox, value) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    def _browse_transcript_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Transcript folder",
            self.transcript_dir.text() or "",
        )
        if directory:
            self.transcript_dir.setText(directory)

    def _update_model_enabled(self) -> None:
        self.model_combo.setEnabled(not self.auto_model.isChecked())
