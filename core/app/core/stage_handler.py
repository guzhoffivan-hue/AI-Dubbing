from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

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

# Optional precondition: a cross-stage rule that must hold before the stage may run.
# It raises to refuse the run (see app.invariants.InvariantError).
Precondition = Callable[["PipelineContext"], None]


@dataclass
class StageHandler:
    run: StageRunner
    artifact: ArtifactLocator | None = None
    inputs: InputLocator | None = None
    precondition: Precondition | None = None


StageHandlerRegistry = dict[PipelineStage, StageHandler]
