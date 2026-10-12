from dataclasses import dataclass


@dataclass
class TranslationConfig:
    source_language: str = "auto"
    target_language: str = "ru"

    preserve_meaning: float = 1.0
    naturalness: float = 0.7
    literary_style: float = 0.5

    use_previous_context: bool = True
    use_next_context: bool = False
    use_scene_context: bool = True
    adapt_cultural_expressions: bool = True


@dataclass
class TranscriptionConfig:
    model: str = "auto"

    detect_language: bool = True
    timestamps: bool = True
    detect_pauses: bool = True
    detect_overlapping_speech: bool = True

    speech_sensitivity: float = 0.5
    min_segment_duration: float = 0.3


@dataclass
class DubbingConfig:
    original_volume: float = 0.25
    voice_volume: float = 1.0
    music_volume: float = 1.0
    effects_volume: float = 1.0

    preserve_background_audio: bool = True
    duck_original_speech: bool = True
    synchronize_speech: bool = True
    adjust_speech_speed: bool = True


@dataclass
class ProjectConfig:
    translation: TranslationConfig
    transcription: TranscriptionConfig
    dubbing: DubbingConfig

    @classmethod
    def default(cls) -> "ProjectConfig":
        return cls(
            translation=TranslationConfig(),
            transcription=TranscriptionConfig(),
            dubbing=DubbingConfig(),
        )