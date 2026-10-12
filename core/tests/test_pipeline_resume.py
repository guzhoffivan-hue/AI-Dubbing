import json

from app.core.pipeline import Pipeline
from app.core.pipeline_context import PipelineContext
from app.core.pipeline_state import PipelineState
from app.core.stage_handler import StageHandler
from app.models.pipeline_stage import PipelineStage
from app.models.stage_state import StageStatus
from projects.project import Project


def make_context(tmp_path) -> PipelineContext:
    source_video = tmp_path / "video.mp4"
    source_video.touch()

    project = Project.create(
        source_video=source_video,
        projects_dir=tmp_path / "projects",
    )

    project.project_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return PipelineContext(project=project)


def test_state_is_persisted_after_stage_completes(tmp_path):
    context = make_context(tmp_path)
    state_path = context.project_dir / "pipeline_state.json"

    pipeline = Pipeline(
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(
                run=lambda ctx: None,
            ),
        },
    )

    pipeline.run()

    assert state_path.exists()

    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["extract_audio"]["status"] == "completed"


def test_state_is_restored_by_new_instance(tmp_path):
    context = make_context(tmp_path)

    pipeline = Pipeline(
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(
                run=lambda ctx: None,
            ),
        },
    )

    pipeline.run()

    restored = PipelineState.load(context.project_dir)

    assert restored.get("extract_audio").status == StageStatus.COMPLETED


def test_running_stage_is_recovered_and_rerun(tmp_path):
    context = make_context(tmp_path)

    state = PipelineState()
    state.get("extract_audio").status = StageStatus.RUNNING
    state.save(context.project_dir)

    # Simulate a fresh application restart.
    recovered = PipelineState.load(context.project_dir)
    assert recovered.get("extract_audio").status == StageStatus.RUNNING

    calls = []

    pipeline = Pipeline(
        state=recovered,
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(
                run=lambda ctx: calls.append(ctx),
            ),
        },
    )

    pipeline.run()

    assert len(calls) == 1
    assert (
        recovered.get("extract_audio").status
        == StageStatus.COMPLETED
    )


def test_handler_error_is_persisted_as_failed(tmp_path):
    context = make_context(tmp_path)

    def boom(ctx):
        raise RuntimeError("kaput")

    pipeline = Pipeline(
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(run=boom),
        },
    )

    pipeline.run()

    restored = PipelineState.load(context.project_dir)
    state = restored.get("extract_audio")

    assert state.status == StageStatus.FAILED
    assert "kaput" in state.message


def test_corrupt_state_file_loads_as_empty(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    (project_dir / "pipeline_state.json").write_text(
        "{ not valid json",
        encoding="utf-8",
    )

    state = PipelineState.load(project_dir)

    assert state.stages == {}
    assert not state.is_completed("extract_audio")


def test_save_failure_is_not_masked_as_success(tmp_path):
    context = make_context(tmp_path)

    # A path that is a file, not a directory: writing state must fail.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    pipeline = Pipeline(
        context=context,
        project_dir=blocked,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(
                run=lambda ctx: None,
            ),
        },
    )

    pipeline.run()

    state = pipeline.state.get("extract_audio")

    assert state.status == StageStatus.FAILED
    assert "State save failed" in state.message


def test_reentrant_run_is_rejected(tmp_path):
    context = make_context(tmp_path)

    def reenter(ctx):
        pipeline.run()

    pipeline = Pipeline(
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(run=reenter),
        },
    )

    pipeline.run()

    state = pipeline.state.get("extract_audio")

    assert state.status == StageStatus.FAILED
    assert "RuntimeError" in state.message
