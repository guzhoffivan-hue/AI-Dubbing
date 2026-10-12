from dataclasses import dataclass


@dataclass
class SpeechSegment:
    id: int
    start: float
    end: float
    speaker: str
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start