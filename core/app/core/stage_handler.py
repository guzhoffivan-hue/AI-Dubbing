from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

from app.core.progress import LineCallback, StageProgressUpdate
from app.models.pipeline_stage import PipelineStage

if TYPE_CHECKING:
    from app.core.pipeline_context import PipelineContext


@dataclass
class StageOutcome:
    """Result of a stage handler.

    ``value`` is stored in the pipeline context; ``artifact`` points to the
    file the stage is expected to produce. When an artifact is declared it
    must exist and be non-empty, otherwise the stage is treated as failed.
    """

    value: object | None = None
    artifact: Path | None = None


# A handler receives the pipeline context and may return an outcome.
StageRunner = Callable[
    ["PipelineContext"],
    Union[StageOutcome, None],
]

# Optional artifact locator used to validate an already completed stage.
ArtifactLocator = Callable[["PipelineContext"], Union[Path, None]]

# Optional input locator: the files this stage consumed. Their fingerprints are recorded
# together with the artifact, so a change in any input invalidates the saved completion.
InputLocator = Callable[["PipelineContext"], list[Path]]

# Optional parameter locator: the settings that shape this stage's output. Their hash
# goes into the artifact manifest, so a model or backend change invalidates exactly the
# stages it affects instead of every completed stage.
ParameterLocator = Callable[["PipelineContext"], Union[dict, None]]

# Optional precondition: a cross-stage rule that must hold before the stage may run.
# It raises to refuse the run (see app.invariants.InvariantError).
Precondition = Callable[["PipelineContext"], None]

# Optional progress reader: receives one log line of the running stage and returns what
# it means for progress, or None when the line says nothing (see app.core.progress). The
# reader keeps its own state, so it may be an object rather than a plain function.
ProgressReader = Callable[[str], Union[StageProgressUpdate, None]]


@dataclass
class StageHandler:
    run: StageRunner
    artifact: ArtifactLocator | None = None
    inputs: InputLocator | None = None
    parameters: ParameterLocator | None = None
    precondition: Precondition | None = None
    # Deadline for one attempt, in seconds (None = no limit). The deadline is
    # per attempt and enforced through the stage's cancel token, so a runner that
    # owns a process can stop it; a blocking handler is still reported as timed
    # out, only once it returns.
    timeout: float | None = None
    # Extra attempts a failing stage gets before the pipeline gives up. A user
    # cancel is never retried, because the user asked for the run to stop.
    retries: int = 0
    # Turns this stage's log lines into progress, so a stage that runs for hours
    # reports where it is instead of looking hung.
    progress: ProgressReader | None = None


StageHandlerRegistry = dict[PipelineStage, StageHandler]
