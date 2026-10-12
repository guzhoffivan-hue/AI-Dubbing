"""Cancellation, timeouts and retries of pipeline stages.

These tests pin the behaviour the GUI depends on: a user cancel stops the run
without pretending the stage finished, a timeout fails only its own stage, and a
failing stage gets its retries while a cancel never does.
"""

import threading
import time

import pytest

from app import adapters
from app.core.cancellation import (
    CancelToken,
    StageCancelled,
    StageTimeout,
)
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
    project.project_dir.mkdir(parents=True, exist_ok=True)

    return PipelineContext(project=project)


def ok_handler(log, name="stage", **kwargs) -> StageHandler:
    def run(_context):
        log.append(name)
        return StageOutcome(value=name)

    return StageHandler(run=run, **kwargs)


# -- the token itself -------------------------------------------------------------------


def test_a_child_token_is_cancelled_by_its_parent():
    parent = CancelToken()
    child = parent.child()

    assert not child.cancelled

    parent.request("user asked to stop")

    assert child.cancelled
    assert child.reason == "user asked to stop"


def test_a_child_cancel_does_not_reach_its_sibling():
    parent = CancelToken()
    first = parent.child()
    second = parent.child()

    first.request("timeout", 5.0)

    assert first.cancelled
    assert not second.cancelled
    assert not parent.cancelled


def test_raise_if_cancelled_keeps_the_reason():
    cancelled = CancelToken()
    cancelled.request("user asked to stop")

    with pytest.raises(StageCancelled) as exc:
        cancelled.raise_if_cancelled()

    assert exc.value.reason == "user asked to stop"

    timed_out = CancelToken()
    timed_out.request("timeout", 12.5)

    with pytest.raises(StageTimeout) as exc:
        timed_out.raise_if_cancelled()

    assert exc.value.seconds == 12.5


def test_a_token_that_was_never_asked_does_nothing():
    CancelToken().raise_if_cancelled()


# -- the pipeline -----------------------------------------------------------------------


def test_cancel_before_a_stage_stops_the_run(tmp_path):
    context = make_context(tmp_path)
    token = CancelToken()
    ran = []

    pipeline = Pipeline(
        context=context,
        cancel=token,
        handlers={PipelineStage.EXTRACT_AUDIO: ok_handler(ran, "extract_audio")},
    )
    token.request("user asked to stop")
    pipeline.run()

    analysis = pipeline.state.get(PipelineStage.EXTRACT_AUDIO.value)

    assert ran == []
    assert analysis.status == StageStatus.CANCELLED
    assert "Cancelled before extract_audio" in analysis.message

    # The run stops instead of walking the remaining stages.
    assert pipeline.state.get(PipelineStage.TRANSCRIPTION.value).message == ""


def test_cancel_during_a_stage_marks_it_cancelled_and_stops(tmp_path):
    context = make_context(tmp_path)
    token = CancelToken()
    ran = []

    def stopper(_context):
        ran.append("extract_audio")
        token.request("user asked to stop")
        return StageOutcome(value="half done")

    pipeline = Pipeline(
        context=context,
        cancel=token,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(run=stopper),
            PipelineStage.TRANSCRIPTION: ok_handler(ran, "transcription"),
        },
    )
    pipeline.run()

    analysis = pipeline.state.get(PipelineStage.EXTRACT_AUDIO.value)

    # The stage returned, but it was stopped on purpose: it is not completed.
    assert ran == ["extract_audio"]
    assert analysis.status == StageStatus.CANCELLED
    assert not pipeline.state.is_completed(PipelineStage.EXTRACT_AUDIO.value)
    assert pipeline.state.get(PipelineStage.TRANSCRIPTION.value).message == ""


def test_a_user_cancel_is_never_retried(tmp_path):
    context = make_context(tmp_path)
    attempts = []

    def stopper(_context):
        attempts.append(1)
        raise StageCancelled("user asked to stop")

    pipeline = Pipeline(
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(run=stopper, retries=5),
        },
    )
    pipeline.run()

    assert len(attempts) == 1
    assert pipeline.state.get(PipelineStage.EXTRACT_AUDIO.value).status == StageStatus.CANCELLED


def test_timeout_fails_the_stage_and_the_run_continues(tmp_path):
    context = make_context(tmp_path)
    finished = []

    def slow(_context):
        time.sleep(0.4)
        return StageOutcome(value="late")

    pipeline = Pipeline(
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(run=slow, timeout=0.05),
            PipelineStage.TRANSCRIPTION: ok_handler(finished, "transcription"),
        },
    )
    pipeline.run()

    analysis = pipeline.state.get(PipelineStage.EXTRACT_AUDIO.value)
    extract = pipeline.state.get(PipelineStage.TRANSCRIPTION.value)

    assert analysis.status == StageStatus.FAILED
    assert "timeout" in analysis.message

    # A timeout fails only its own stage: the run keeps going and the dependent
    # stage is blocked, not cancelled, so the run can resume later.
    assert extract.status == StageStatus.NOT_STARTED
    assert "Dependencies not completed" in extract.message
    assert finished == []


def test_a_timed_out_stage_is_retried(tmp_path):
    context = make_context(tmp_path)
    attempts = []

    def slow(_context):
        attempts.append(1)
        time.sleep(0.3)
        return StageOutcome(value="late")

    pipeline = Pipeline(
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(run=slow, timeout=0.05, retries=2),
        },
    )
    pipeline.run()

    assert len(attempts) == 3
    assert pipeline.state.get(PipelineStage.EXTRACT_AUDIO.value).status == StageStatus.FAILED


def test_retries_let_a_flaky_stage_finish(tmp_path):
    context = make_context(tmp_path)
    attempts = []

    def flaky(_context):
        attempts.append(1)

        if len(attempts) < 3:
            raise RuntimeError("transient boom")

        return StageOutcome(value="ok")

    pipeline = Pipeline(
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(run=flaky, retries=2),
        },
    )
    pipeline.run()

    analysis = pipeline.state.get(PipelineStage.EXTRACT_AUDIO.value)

    assert len(attempts) == 3
    assert analysis.status == StageStatus.COMPLETED
    assert analysis.progress == 1.0


def test_without_retries_a_failure_stays_a_failure(tmp_path):
    context = make_context(tmp_path)
    attempts = []

    def broken(_context):
        attempts.append(1)
        raise RuntimeError("boom")

    pipeline = Pipeline(
        context=context,
        handlers={PipelineStage.EXTRACT_AUDIO: StageHandler(run=broken)},
    )
    pipeline.run()

    assert len(attempts) == 1
    assert pipeline.state.get(PipelineStage.EXTRACT_AUDIO.value).status == StageStatus.FAILED


def test_the_deadline_is_per_attempt(tmp_path):
    context = make_context(tmp_path)
    durations = []

    def slow(_context):
        started = time.monotonic()
        time.sleep(0.15)
        durations.append(time.monotonic() - started)
        return StageOutcome(value="ok")

    pipeline = Pipeline(
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: StageHandler(run=slow, timeout=1.0, retries=1),
        },
    )
    pipeline.run()

    # A generous deadline must not fire on the second attempt just because the
    # first one used it.
    assert len(durations) == 1
    assert pipeline.state.get(PipelineStage.EXTRACT_AUDIO.value).status == StageStatus.COMPLETED


def test_the_context_carries_the_stage_token(tmp_path):
    context = make_context(tmp_path)
    seen = []

    def watch(run_context):
        seen.append(run_context.cancel)
        return StageOutcome(value="ok")

    pipeline = Pipeline(
        context=context,
        handlers={PipelineStage.EXTRACT_AUDIO: StageHandler(run=watch)},
    )
    pipeline.run()

    assert len(seen) == 1
    assert seen[0] is not None
    # The token is gone once the stage is over: a stale token must not be polled.
    assert context.cancel is None


def test_cancelled_status_survives_a_reload(tmp_path):
    context = make_context(tmp_path)
    state = PipelineState()
    state.get(PipelineStage.EXTRACT_AUDIO.value).status = StageStatus.CANCELLED
    state.save(context.project_dir)

    restored = PipelineState.load(context.project_dir)

    assert restored.get(PipelineStage.EXTRACT_AUDIO.value).status == StageStatus.CANCELLED
    assert not restored.is_completed(PipelineStage.EXTRACT_AUDIO.value)


def test_request_cancel_stops_the_next_stage(tmp_path):
    context = make_context(tmp_path)
    ran = []

    pipeline = Pipeline(
        context=context,
        handlers={
            PipelineStage.EXTRACT_AUDIO: ok_handler(ran, "extract_audio"),
            PipelineStage.TRANSCRIPTION: ok_handler(ran, "transcription"),
        },
    )

    # A GUI calls this from another thread; the pipeline stops between stages.
    def cancel_from_another_thread():
        pipeline.request_cancel("user closed the window")

    worker = threading.Thread(target=cancel_from_another_thread)
    worker.start()
    worker.join()
    pipeline.run()

    assert ran == []
    assert pipeline.state.get(PipelineStage.EXTRACT_AUDIO.value).status == StageStatus.CANCELLED


# -- the runner that owns the process ----------------------------------------------------


def test_subprocess_runner_stops_a_running_stage_process(tmp_path):
    token = CancelToken()
    script = tmp_path / "stage.py"
    script.write_text("import time\n\ntime.sleep(30)\nprint('never printed')\n")
    log_path = tmp_path / "stage.log"

    timer = threading.Timer(0.4, token.request, args=("user asked to stop",))
    timer.daemon = True
    timer.start()

    started = time.monotonic()

    with pytest.raises(StageCancelled):
        adapters.subprocess_runner(
            ["python3", str(script)],
            {},
            tmp_path,
            log_path,
            cancel=token,
        )

    elapsed = time.monotonic() - started
    timer.cancel()

    # 30 s of sleep must not be waited out: the process is stopped instead.
    assert elapsed < 15
    text = log_path.read_text()
    assert "stage stopped" in text
    assert "never printed" not in text


def test_subprocess_runner_reports_a_timeout_as_such(tmp_path):
    token = CancelToken()
    script = tmp_path / "stage.py"
    script.write_text("import time\n\ntime.sleep(30)\n")
    log_path = tmp_path / "stage.log"

    timer = threading.Timer(0.3, token.request, args=("timeout", 0.3))
    timer.daemon = True
    timer.start()

    with pytest.raises(StageTimeout):
        adapters.subprocess_runner(
            ["python3", str(script)],
            {},
            tmp_path,
            log_path,
            cancel=token,
        )

    timer.cancel()
    assert "stage stopped (timeout)" in log_path.read_text()


def test_subprocess_runner_without_a_token_still_reports_failures(tmp_path):
    script = tmp_path / "stage.py"
    script.write_text("import sys\n\nprint('bad', file=sys.stderr)\nsys.exit(3)\n")
    log_path = tmp_path / "stage.log"

    with pytest.raises(adapters.AdapterError) as exc:
        adapters.subprocess_runner(["python3", str(script)], {}, tmp_path, log_path)

    assert "exit code 3" in str(exc.value)
