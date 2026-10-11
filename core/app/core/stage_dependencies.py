from app.models.pipeline_stage import PipelineStage


STAGE_DEPENDENCIES: dict[
    PipelineStage,
    list[PipelineStage],
] = {
    PipelineStage.ANALYSIS: [],

    PipelineStage.TRANSCRIPTION: [
        PipelineStage.ANALYSIS,
    ],

    PipelineStage.DIARIZATION: [
        PipelineStage.TRANSCRIPTION,
    ],

    PipelineStage.TRANSLATION: [
        PipelineStage.TRANSCRIPTION,
        PipelineStage.DIARIZATION,
    ],

    PipelineStage.VOICE_ASSIGNMENT: [
        PipelineStage.TRANSLATION,
    ],

    PipelineStage.TTS: [
        PipelineStage.VOICE_ASSIGNMENT,
    ],

    PipelineStage.AUDIO_MIX: [
        PipelineStage.TTS,
    ],

    PipelineStage.VIDEO_RENDER: [
        PipelineStage.AUDIO_MIX,
    ],
}