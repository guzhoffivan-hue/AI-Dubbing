from enum import Enum


class PipelineStage(str, Enum):
    EXTRACT_AUDIO = "extract_audio"
    TRANSCRIPTION = "transcription"
    ALIGNMENT = "alignment"
    DIARIZATION = "diarization"
    SEGMENT_MERGE = "segment_merge"
    TRANSLATION = "translation"
    DIALOGUE = "dialogue"
    TTS = "tts"
    SFX = "sfx"
    AMBIENCE = "ambience"
    AUDIO_MIX = "audio_mix"
    VIDEO_RENDER = "video_render"