#!/usr/bin/env python3
"""End-to-end engine run driven by the core adapters.

Runs all twelve engine stages through core/app/adapters on a cut of the source
clip, with the pipeline state and the artifact manifest saved next to the work
dir so a re-run resumes instead of repeating finished stages.

Run from anywhere:  /opt/homebrew/bin/python3 core/e2e_run.py [duration]

The guard refuses to run unless the work dir is fresh (it must never touch the
live project's artifacts -- see test_run_1200s/preflight/INCIDENT_wrong_root).

Ctrl-C cancels instead of killing the process: the running stage stops its own
subprocess and the state keeps which stage was interrupted, so ``--resume``
continues from there.
"""
from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parent
ROOT = CORE.parent
sys.path.insert(0, str(CORE))

from app.adapters import EngineLayout, make_handlers  # noqa: E402
from app.core.cancellation import CancelToken  # noqa: E402
from app.core.pipeline import Pipeline  # noqa: E402
from app.core.pipeline_state import PipelineState  # noqa: E402
from app.core.stage_handler import StageHandler, StageOutcome  # noqa: E402
from app.models.pipeline_stage import PipelineStage  # noqa: E402

CLIP = ROOT / "input" / "42503442096-1-192.mp4"


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: e2e_run.py [--resume] [duration_seconds]\n"
              "  Runs all 12 engine stages via core adapters.\n"
              "  Example: python3 core/e2e_run.py 150.0", flush=True)
        return 0
    args = [a for a in sys.argv[1:] if a != "--resume"]
    resume = "--resume" in sys.argv
    duration = float(args[0]) if args else 150.0
    work = ROOT / "work" / f"core_e2e_{int(duration)}s"

    if not CLIP.exists():
        print(f"GUARD: source clip missing: {CLIP}", flush=True)
        return 2
    if not (ROOT / "scripts" / "cli.py").exists():
        print(f"GUARD: {ROOT} does not look like the engine root", flush=True)
        return 2
    if work.exists() and any(work.iterdir()) and not resume:
        print(f"GUARD: {work} already exists and is not empty; "
              "remove it, pick another duration, or pass --resume", flush=True)
        return 2

    work.mkdir(parents=True, exist_ok=True)
    layout = EngineLayout(
        root=ROOT,
        work_dir=work,
        input_video=CLIP,
        output_dir=ROOT / "output" / f"core_e2e_{int(duration)}s",
        start=0.0,
        duration=duration,
    )
    layout.check_config()

    handlers = make_handlers(layout)


    from uuid import uuid4
    from projects.project import Project
    from app.core.pipeline_context import PipelineContext
    from app.models.project_config import ProjectConfig

    project_dir = work / "_core_state"
    project_dir.mkdir(exist_ok=True)
    if resume and (project_dir / "pipeline_state.json").exists():
        state = PipelineState.load(project_dir)
        print(f"resume: loaded state from {project_dir}", flush=True)
    else:
        state = PipelineState()
    context = PipelineContext(
        project=Project(
            id=uuid4(),
            name=CLIP.stem,
            source_video=CLIP,
            project_dir=project_dir,
            config=ProjectConfig.default(),
        ),
    )

    cancel = CancelToken()

    def on_interrupt(_signum, _frame) -> None:
        # First Ctrl-C asks the running stage to stop; a second one is left to the
        # default handler, so an unresponsive stage can still be killed by hand.
        if cancel.cancelled:
            raise KeyboardInterrupt

        print("\ncancel requested: stopping the running stage", flush=True)
        cancel.request("Ctrl-C")

    signal.signal(signal.SIGINT, on_interrupt)

    def report(progress) -> None:
        # A stage reports its own progress from its log while it runs, so the
        # line is worth printing as-is instead of a bare percentage.
        print(
            f"[{progress.stage.value}] {progress.progress:>4.0%} {progress.message}",
            flush=True,
        )

    pipe = Pipeline(
        state=state,
        context=context,
        handlers=handlers,
        on_progress=report,
        cancel=cancel,
        project_dir=project_dir,
    )

    import time
    started = time.monotonic()
    pipe.run(context)

    print("\n=== stage summary ===", flush=True)
    failed = False
    for stage in PipelineStage:
        entry = state.get(stage.value)
        print(f"{stage.value:>18}: {entry.status.value:>10}  {entry.message}", flush=True)
        failed |= entry.status.value == "failed"

    final = layout.final_file
    if final.exists():
        seconds = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(final)],
            text=True,
        ).strip()
        print(f"\nFINAL: {final} ({final.stat().st_size} bytes, {seconds}s)", flush=True)
    print(f"elapsed: {time.monotonic() - started:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
