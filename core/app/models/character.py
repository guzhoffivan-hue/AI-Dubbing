from dataclasses import dataclass


@dataclass
class Character:
    id: str
    name: str
    speaker_id: str
    voice_id: str | None = None

    volume: float = 1.0
    speed: float = 1.0
    pitch: float = 0.0