from app.core.pipeline_state import PipelineState
from app.models.stage_state import StageStatus


def test_pipeline_state_save_and_load(tmp_path):
    state = PipelineState()

    analysis = state.get("analysis")

    analysis.status = StageStatus.COMPLETED
    analysis.progress = 1.0
    analysis.message = "Media analysis completed"

    path = state.save(tmp_path)

    assert path.exists()
    assert path == tmp_path / "pipeline_state.json"

    loaded = PipelineState.load(tmp_path)

    assert loaded.is_completed("analysis")

    loaded_analysis = loaded.get("analysis")

    assert loaded_analysis.stage == "analysis"
    assert loaded_analysis.status == StageStatus.COMPLETED
    assert loaded_analysis.progress == 1.0
    assert loaded_analysis.message == "Media analysis completed"


def test_pipeline_state_default_stage():
    state = PipelineState()

    stage = state.get("translation")

    assert stage.stage == "translation"
    assert stage.status == StageStatus.NOT_STARTED
    assert stage.progress == 0.0
    assert stage.message == ""
    