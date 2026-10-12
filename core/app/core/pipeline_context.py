from dataclasses import dataclass, field
from pathlib import Path

from app.core.cancellation import CancelToken
from app.core.progress import LineCallback
from projects.project import Project


@dataclass
class PipelineContext:
    project: Project

    metadata: dict = field(default_factory=dict)
    transcript: object | None = None
    translation: object | None = None
    characters: list = field(default_factory=list)
    # Set by the pipeline while a stage runs: the runner polls this token so the
    # stage can stop its own process on a user cancel or a timeout.
    cancel: CancelToken | None = None
    # Set by the pipeline while a stage runs: fed with every log line of the
    # stage, so its progress reader can report how far the stage has got.
    on_output_line: LineCallback | None = None

    @property
    def project_dir(self) -> Path:
        return self.project.project_dir

    @property
    def source_video(self) -> Path:
        return self.project.source_video

    @property
    def analysis_dir(self) -> Path:
        return self.project_dir / "analysis"

    @property
    def transcription_dir(self) -> Path:
        return self.project_dir / "transcription"

    @property
    def translation_dir(self) -> Path:
        return self.project_dir / "translation"

    @property
    def voices_dir(self) -> Path:
        return self.project_dir / "voices"

    @property
    def audio_dir(self) -> Path:
        return self.project_dir / "audio"

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "output"