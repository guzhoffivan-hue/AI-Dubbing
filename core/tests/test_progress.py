"""Progress inside a long stage.

A stage that runs for hours must say where it is: these tests pin that a log line
becomes progress, that the state carries it while the stage runs, and that a
broken reader can never fail a stage that is working.
"""

import pytest

from app import adapters
from app.core.pipeline import Pipeline
from app.core.pipeline_context import PipelineContext
from app.core.progress import LogProgress
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


ASR_PATTERNS = (r"ASR Chunk (\d+)/(\d+) completed",)


# -- the reader --------------------------------------------------------------------------


def test_a_line_with_a_total_reports_its_fraction():
    reader = LogProgress(ASR_PATTERNS)

    update = reader("ASR Chunk 2/4 completed (975 chars)")

    assert update is not None
    assert update.fraction == 0.5
    assert "ASR Chunk 2/4" in update.message


def test_the_stage_can_announce_its_own_total():
    reader = LogProgress(
        ASR_PATTERNS,
        total_pattern=r"Splitting into (\d+) safe chunks",
    )

    assert reader("Splitting into 8 safe chunks (120s each)...") is None
    assert reader("ASR Chunk 1/8 completed (975 chars)").fraction == pytest.approx(0.125)


def test_a_bare_counter_uses_the_total_from_the_input():
    reader = LogProgress((r"segment (\d+) speaker=",), total=10)

    assert reader("segment 002 speaker=SPEAKER_00").fraction == pytest.approx(0.2)


def test_a_counter_without_any_total_is_not_progress():
    reader = LogProgress((r"segment (\d+) speaker=",))

    assert reader("segment 004 speaker=SPEAKER_00") is None


def test_a_line_that_says_nothing_is_not_progress():
    reader = LogProgress(ASR_PATTERNS, total_pattern=r"Splitting into (\d+) safe chunks")

    assert reader("") is None
    assert reader("Loading checkpoint shards:  50%|#####     | 1/2") is None
    assert reader("Setting `pad_token_id` to `eos_token_id`:151645") is None


def test_progress_never_leaves_the_zero_to_one_range():
    reader = LogProgress((r"line (\d+)/(\d+)",))

    assert reader("line 9/8").fraction == 1.0
    assert reader("line 0/8").fraction == 0.0


# -- the pipeline ------------------------------------------------------------------------


def test_a_stage_reports_progress_while_it_runs(tmp_path):
    context = make_context(tmp_path)
    reports = []
    progress_inside_the_stage = []

    def run(run_context):
        run_context.on_output_line("ASR Chunk 1/2 completed (975 chars)")
        progress_inside_the_stage.append(
            pipeline.state.get(PipelineStage.ANALYSIS.value).progress
        )
        run_context.on_output_line("ASR Chunk 2/2 completed (1008 chars)")
        return StageOutcome(value="ok")

    pipeline = Pipeline(
        context=context,
        on_progress=reports.append,
        handlers={},
    )
    pipeline.register(
        PipelineStage.ANALYSIS,
        StageHandler(run=run, progress=LogProgress(ASR_PATTERNS)),
    )
    pipeline.run()

    fractions = [
        report.progress for report in reports if report.stage == PipelineStage.ANALYSIS
    ]

    assert 0.5 in fractions
    # The state carries the live value while the stage runs, without rewriting the
    # state file for every line.
    assert progress_inside_the_stage == [0.5]
    # The finished stage still ends at 1.0 and is reported as completed.
    assert pipeline.state.get(PipelineStage.ANALYSIS.value).status == StageStatus.COMPLETED
    assert fractions[-1] == 1.0


def test_a_stage_without_a_reader_gets_no_intermediate_reports(tmp_path):
    context = make_context(tmp_path)
    reports = []

    def run(run_context):
        run_context.on_output_line("anything at all")

        return StageOutcome(value="ok")

    pipeline = Pipeline(context=context, on_progress=reports.append, handlers={})
    pipeline.register(PipelineStage.ANALYSIS, StageHandler(run=run))
    pipeline.run()

    assert [
        report.progress for report in reports if report.stage == PipelineStage.ANALYSIS
    ] == [0.0, 1.0]


def test_a_broken_reader_cannot_fail_a_working_stage(tmp_path):
    context = make_context(tmp_path)
    reports = []

    def broken_reader(_line):
        raise ValueError("this reader is wrong")

    def run(run_context):
        run_context.on_output_line("whatever")

        return StageOutcome(value="ok")

    pipeline = Pipeline(context=context, on_progress=reports.append, handlers={})
    pipeline.register(
        PipelineStage.ANALYSIS,
        StageHandler(run=run, progress=broken_reader),
    )
    pipeline.run()

    assert pipeline.state.get(PipelineStage.ANALYSIS.value).status == StageStatus.COMPLETED


def test_the_line_callback_is_cleared_after_the_stage(tmp_path):
    context = make_context(tmp_path)

    def run(run_context):
        return StageOutcome(value="ok")

    pipeline = Pipeline(
        context=context,
        handlers={PipelineStage.ANALYSIS: StageHandler(run=run)},
    )
    pipeline.run()

    assert context.on_output_line is None


# -- the runner that owns the process ----------------------------------------------------


def test_the_runner_streams_output_lines_as_they_arrive(tmp_path):
    script = tmp_path / "stage.py"
    script.write_text(
        "import time\n"
        "for i in range(1, 4):\n"
        "    print(f'unit {i}/3 done', flush=True)\n"
        "    time.sleep(0.05)\n"
    )
    log_path = tmp_path / "stage.log"
    seen: list[str] = []

    adapters.subprocess_runner(
        ["python3", str(script)],
        {},
        tmp_path,
        log_path,
        on_line=seen.append,
    )

    assert [line.strip() for line in seen] == [
        "unit 1/3 done",
        "unit 2/3 done",
        "unit 3/3 done",
    ]
    # The log is written while the stage runs, so a crashed stage still has its log.
    assert "unit 3/3 done" in log_path.read_text()
