from dataclasses import dataclass, field

from .character import Character
from .speech_segment import SpeechSegment


@dataclass
class Transcript:
    segments: list[SpeechSegment] = field(default_factory=list)
    characters: list[Character] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if not self.segments:
            return 0.0

        return max(segment.end for segment in self.segments)

    def add_segment(self, segment: SpeechSegment) -> None:
        self.segments.append(segment)

    def add_character(self, character: Character) -> None:
        self.characters.append(character)