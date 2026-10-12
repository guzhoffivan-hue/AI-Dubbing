from app.models.pipeline_stage import PipelineStage


STAGE_DEPENDENCIES: dict[
    PipelineStage,
    list[PipelineStage],
] = {
    PipelineStage.EXTRACT_AUDIO: [],

    PipelineStage.TRANSCRIPTION: [
        PipelineStage.EXTRACT_AUDIO,
    ],

    PipelineStage.ALIGNMENT: [
        PipelineStage.TRANSCRIPTION,
    ],

    PipelineStage.DIARIZATION: [
        PipelineStage.TRANSCRIPTION,
    ],

    PipelineStage.SEGMENT_MERGE: [
        PipelineStage.ALIGNMENT,
        PipelineStage.DIARIZATION,
    ],

    PipelineStage.TRANSLATION: [
        PipelineStage.SEGMENT_MERGE,
    ],

    PipelineStage.DIALOGUE: [
        PipelineStage.TRANSLATION,
    ],

    PipelineStage.TTS: [
        PipelineStage.DIALOGUE,
    ],

    PipelineStage.SFX: [
        PipelineStage.TRANSLATION,
    ],

    PipelineStage.AMBIENCE: [
        PipelineStage.TRANSLATION,
    ],

    PipelineStage.AUDIO_MIX: [
        PipelineStage.TTS,
        PipelineStage.SFX,
        PipelineStage.AMBIENCE,
    ],

    PipelineStage.VIDEO_RENDER: [
        PipelineStage.AUDIO_MIX,
    ],
}