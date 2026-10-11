import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.artifact_manifest import (
    ArtifactStore,
    build_manifest,
    verify_manifest,
)
from app.core.pipeline_context import PipelineContext
from app.core.pipeline_state import PipelineState
from app.core.stage_dependencies import STAGE_DEPENDENCIES
from app.core.stage_handler import (
    StageHandler,
    StageHandlerRegistry,
    StageOutcome,
)
from app.models.pipeline_stage import PipelineStage
from app.models.stage_state import StageState, StageStatus


@dataclass
class PipelineProgress:
    stage: PipelineStage
    progress: float
    message: str


class Pipeline:
    def __init__(
        self,
        state: PipelineState | None = None,
        context: PipelineContext | None = None,
        handlers: StageHandlerRegistry | None = None,
        on_progress: Callable[[PipelineProgress], None] | None = None,
        project_dir: Path | None = None,
        artifact_store: ArtifactStore | None = None,
    ):
        self.state = state or PipelineState()
        self.context = context
        self.handlers: StageHandlerRegistry = dict(
            handlers or {}
        )
        self.on_progress = on_progress
        self.results: dict[str, object] = {}

        if project_dir is not None:
            self.project_dir = project_dir
        elif context is not None:
            self.project_dir = context.project_dir
        else:
            self.project_dir = None

        if artifact_store is not None:
            self.artifact_store = artifact_store
        elif self.project_dir is not None:
            self.artifact_store = ArtifactStore.load(
                self.project_dir
            )
        else:
            self.artifact_store = ArtifactStore()

        self._is_running = False

    def register(
        self,
        stage: PipelineStage,
        handler: StageHandler,
    ) -> None:
        self.handlers[stage] = handler

    def report(
        self,
        stage: PipelineStage,
        progress: float,
        message: str,
    ) -> None:
        if self.on_progress is not None:
            self.on_progress(
                PipelineProgress(
                    stage=stage,
                    progress=progress,
                    message=message,
                )
            )

    def dependencies_completed(
        self,
        stage: PipelineStage,
    ) -> bool:
        dependencies = STAGE_DEPENDENCIES.get(
            stage,
            [],
        )

        return all(
            self.state.is_completed(
                dependency.value
            )
            for dependency in dependencies
        )

    @staticmethod
    def artifact_is_valid(path: Path | None) -> bool:
        if path is None:
            return True

        return path.exists() and path.stat().st_size > 0

    @staticmethod
    def _declared_inputs(
        handler: StageHandler | None,
        context: PipelineContext | None,
    ) -> list[Path]:
        if (
            handler is None
            or handler.inputs is None
            or context is None
        ):
            return []

        try:
            return list(handler.inputs(context) or [])
        except Exception:
            # An input locator that cannot answer must never make a stage look
            # valid, so report "no inputs recorded" rather than raising.
            return []

    @staticmethod
    def _parameters(
        context: PipelineContext | None,
    ) -> dict:
        config = getattr(context, "config", None)

        if config is None:
            return {}

        try:
            return json.loads(
                json.dumps({"config": config}, default=str)
            )
        except (TypeError, ValueError):
            return {"config": str(config)}

    def _completed_stage_is_still_valid(
        self,
        handler: StageHandler | None,
        context: PipelineContext | None,
        stage: PipelineStage | None = None,
    ) -> bool:
        """A completed stage may be skipped only when it can be trusted.

        A stage without a handler is never trusted (nothing to validate a
        saved completion against). When the handler declares an artifact,
        the artifact must exist and be non-empty; and once a manifest has
        been recorded for the stage, the artifact and every declared input
        must still match it and the run parameters must be unchanged.
        """

        if handler is None:
            return False

        if handler.artifact is None:
            return True

        if context is None:
            return False

        if not self.artifact_is_valid(handler.artifact(context)):
            return False

        if stage is not None:
            manifest = self.artifact_store.get(stage.value)

            if manifest is not None:
                ok, _reason = verify_manifest(
                    manifest,
                    inputs=self._declared_inputs(
                        handler,
                        context,
                    ),
                    parameters=self._parameters(context),
                )

                return ok

        return True

    def _persist(self) -> None:
        if self.project_dir is None:
            return

        self.state.save(self.project_dir)

    def _apply(
        self,
        stage: PipelineStage,
        stage_state: StageState,
        status: StageStatus,
        progress: float,
        message: str,
    ) -> None:
        stage_state.status = status
        stage_state.progress = progress
        stage_state.message = message

        self.report(stage, progress, message)

        try:
            self._persist()
        except Exception as exc:
            # A failed save must never be hidden behind a successful stage.
            stage_state.status = StageStatus.FAILED
            stage_state.progress = 0.0
            stage_state.message = (
                f"State save failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self.report(stage, 0.0, stage_state.message)

    def _ensure_outcome(self, outcome: StageOutcome | None) -> None:
        if outcome is None or outcome.artifact is None:
            return

        if not self.artifact_is_valid(outcome.artifact):
            raise RuntimeError(
                "Expected artifact is missing or empty: "
                f"{outcome.artifact}"
            )

    def _store_result(
        self,
        stage: PipelineStage,
        value: object,
    ) -> None:
        self.results[stage.value] = value

        if self.context is not None:
            self.context.metadata[stage.value] = value

    def run(
        self,
        context: PipelineContext | None = None,
    ) -> None:
        if self._is_running:
            raise RuntimeError(
                "Pipeline.run() is already running on this instance"
            )

        self._is_running = True

        try:
            self._run_stages(context or self.context)
        finally:
            self._is_running = False

    def _run_stages(
        self,
        run_context: PipelineContext | None,
    ) -> None:
        for stage in PipelineStage:
            handler = self.handlers.get(stage)
            stage_state = self.state.get(stage.value)

            # A stage left RUNNING by a previous crash is recoverable: treat
            # it as not started so the pipeline can safely re-run it.
            if stage_state.status == StageStatus.RUNNING:
                self._apply(
                    stage,
                    stage_state,
                    StageStatus.NOT_STARTED,
                    0.0,
                    f"Recovered interrupted {stage.value}",
                )

            if stage_state.status == StageStatus.COMPLETED:
                if self._completed_stage_is_still_valid(
                    handler,
                    run_context,
                    stage,
                ):
                    self.report(
                        stage=stage,
                        progress=1.0,
                        message=f"Skipping {stage.value}",
                    )
                    continue

                # The saved completion cannot be trusted: rerun the stage.
                self._apply(
                    stage,
                    stage_state,
                    StageStatus.NOT_STARTED,
                    0.0,
                    f"Saved completion for {stage.value} is not "
                    f"valid; rerunning",
                )

            if handler is None:
                # A stage without a handler must never be reported as done.
                self._apply(
                    stage,
                    stage_state,
                    StageStatus.NOT_STARTED,
                    0.0,
                    f"No handler registered for {stage.value}",
                )
                continue

            if not self.dependencies_completed(stage):
                self._apply(
                    stage,
                    stage_state,
                    StageStatus.NOT_STARTED,
                    0.0,
                    f"Dependencies not completed for {stage.value}",
                )
                continue

            self._apply(
                stage,
                stage_state,
                StageStatus.RUNNING,
                0.0,
                f"Starting {stage.value}",
            )

            try:
                if handler.precondition is not None:
                    # Cross-stage invariants are checked before the stage is
                    # allowed to touch anything: a stale artifact must stop the
                    # pipeline, not silently produce a broken video.
                    handler.precondition(run_context)

                outcome = handler.run(run_context)
                self._ensure_outcome(outcome)
            except Exception as exc:
                self._apply(
                    stage,
                    stage_state,
                    StageStatus.FAILED,
                    0.0,
                    f"{type(exc).__name__}: {exc}",
                )
                continue

            if outcome is not None and outcome.value is not None:
                self._store_result(stage, outcome.value)

            artifact = (
                outcome.artifact
                if outcome is not None
                and outcome.artifact is not None
                else (
                    handler.artifact(run_context)
                    if handler.artifact is not None
                    and run_context is not None
                    else None
                )
            )

            manifest = build_manifest(
                artifact,
                inputs=self._declared_inputs(
                    handler,
                    run_context,
                ),
                parameters=self._parameters(run_context),
            )

            if manifest is not None:
                self.artifact_store.record(
                    stage.value,
                    manifest,
                )

                try:
                    self.artifact_store.save()
                except OSError:
                    # An unsaved manifest only costs a re-run later; it must
                    # never turn a finished stage into a failed one.
                    pass

            self._apply(
                stage,
                stage_state,
                StageStatus.COMPLETED,
                1.0,
                f"Finished {stage.value}",
            )
