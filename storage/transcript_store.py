from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from speech.types import TranscriptEvent


@dataclass(frozen=True)
class TranscriptRecord:
    speaker: str
    timestamp: str
    text: str
    source: str


class TranscriptStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.json_path = self.directory / f"transcript-{session_id}.json"
        self.txt_path = self.directory / f"transcript-{session_id}.txt"
        self.records: list[TranscriptRecord] = []

    def add(self, event: TranscriptEvent) -> TranscriptRecord:
        record = TranscriptRecord(
            speaker=event.speaker,
            timestamp=datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S"),
            text=event.text,
            source=event.source,
        )
        self.records.append(record)
        return record

    def clear(self) -> None:
        self.records.clear()

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.json_path.write_text(
            json.dumps([asdict(record) for record in self.records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.txt_path.write_text(self._to_text(), encoding="utf-8")

    def _to_text(self) -> str:
        return "\n\n".join(
            f"[{record.timestamp}] {record.speaker}:\n{record.text}" for record in self.records
        )
