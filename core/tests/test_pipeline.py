from app.core.pipeline import Pipeline
from app.core.pipeline_context import PipelineContext
from app.core.pipeline_state import PipelineState
from app.core.stage_handler import StageHandler, StageOutcome
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


def test_unregistered_stage_is_not_completed(tmp_path):
    context = make_context(tmp_path)

    pipeline = Pipeline(context=context)

    pipeline.run()

    for stage in PipelineStage:
        state = pipeline.state.get(stage.value)

        assert state.status == StageStatus.NOT_STARTED
        assert not pipeline.state.is_completed(stage.value)
        assert "No handler registered" in state.message


def test_registered_handler_runs_and_receives_context(tmp_path):
    context = make_context(tmp_path)
    received = []

    handler = StageHandler(
        run=lambda ctx: received.append(ctx),
    )

    pipeline = Pipeline(
        context=context,
        handlers={PipelineStage.EXTRACT_AUDIO: handler},
    )

    pipeline.run()

    assert received == [context]
    assert (
        pipeline.state.get("extract_audio").status
        == StageStatus.COMPLETED
    )


def test_successful_stage_is_completed_and_stores_result(tmp_path):
    context = make_context(tmp_path)

    handler = StageHandler(
        run=lambda ctx: StageOutcome(value="extract_audio-done"),
    )

    pipeline = Pipeline(
        context=context,
        handlers={PipelineStage.EXTRACT_AUDIO: handler},
    )

    pipeline.run()

    assert (
        pipeline.state.get("extract_audio").status
        == StageStatus.COMPLETED
    )
    assert pipeline.results["extract_audio"] == "extract_audio-done"
    assert context.metadata["extract_audio"] == "extract_audio-done"


def test_handler_error_results_in_failed(tmp_path):
    context = make_context(tmp_path)

    def boom(ctx):
        raise ValueError("boom")

    pipeline = Pipeline(
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(run=boom),
        },
    )

    pipeline.run()

    state = pipeline.state.get("extract_audio")

    assert state.status == StageStatus.FAILED
    assert "ValueError" in state.message
    assert "boom" in state.message


def test_dependent_stage_is_not_run_when_dependency_unfinished(
    tmp_path,
):
    context = make_context(tmp_path)
    called = []

    handlers = {
        PipelineStage.TRANSCRIPTION: StageHandler(
            run=lambda ctx: called.append(ctx),
        ),
    }

    # EXTRACT_AUDIO has no handler, so it stays NOT_STARTED.
    pipeline = Pipeline(
        context=context,
        handlers=handlers,
    )

    pipeline.run()

    state = pipeline.state.get("transcription")

    assert state.status == StageStatus.NOT_STARTED
    assert called == []
    assert "Dependencies not completed" in state.message


def test_rerun_does_not_skip_stage_with_missing_artifact(tmp_path):
    state = PipelineState()
    state.get("extract_audio").status = StageStatus.COMPLETED

    artifact = tmp_path / "extract_audio_out.txt"
    calls = []

    def run(ctx):
        calls.append(ctx)
        artifact.write_text("done")
        return StageOutcome(artifact=artifact)

    handler = StageHandler(
        run=run,
        artifact=lambda ctx: artifact,
    )

    pipeline = Pipeline(
        state=state,
        context=make_context(tmp_path),
        handlers={PipelineStage.EXTRACT_AUDIO: handler},
    )

    pipeline.run()

    assert len(calls) == 1
    assert artifact.exists()
    assert state.get("extract_audio").status == StageStatus.COMPLETED


def test_completed_stage_with_valid_artifact_is_skipped(tmp_path):
    state = PipelineState()
    state.get("extract_audio").status = StageStatus.COMPLETED

    artifact = tmp_path / "extract_audio_out.txt"
    artifact.write_text("done")

    calls = []

    handler = StageHandler(
        run=lambda ctx: calls.append(ctx),
        artifact=lambda ctx: artifact,
    )

    pipeline = Pipeline(
        state=state,
        context=make_context(tmp_path),
        handlers={PipelineStage.EXTRACT_AUDIO: handler},
    )

    pipeline.run()

    assert calls == []
    assert state.get("extract_audio").status == StageStatus.COMPLETED


def test_declared_artifact_missing_results_in_failed(tmp_path):
    context = make_context(tmp_path)
    missing = tmp_path / "never_created.txt"

    handler = StageHandler(
        run=lambda ctx: StageOutcome(artifact=missing),
    )

    pipeline = Pipeline(
        context=context,
        handlers={PipelineStage.EXTRACT_AUDIO: handler},
    )

    pipeline.run()

    state = pipeline.state.get("extract_audio")

    assert state.status == StageStatus.FAILED
    assert "artifact" in state.message.lower()


def test_error_is_preserved_after_save_and_load(tmp_path):
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

    pipeline.state.save(tmp_path)

    loaded = PipelineState.load(tmp_path)
    state = loaded.get("extract_audio")

    assert state.status == StageStatus.FAILED
    assert "kaput" in state.message


def test_all_registered_stages_complete(tmp_path):
    context = make_context(tmp_path)

    handlers = {
        stage: StageHandler(run=lambda ctx: None)
        for stage in PipelineStage
    }

    pipeline = Pipeline(
        context=context,
        handlers=handlers,
    )

    pipeline.run()

    for stage in PipelineStage:
        assert pipeline.state.is_completed(stage.value)
