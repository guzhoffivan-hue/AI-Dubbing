from dataclasses import dataclass, field
from pathlib import Path

from projects.project import Project


@dataclass
class PipelineContext:
    project: Project

    metadata: dict = field(default_factory=dict)
    transcript: object | None = None
    translation: object | None = None
    characters: list = field(default_factory=list)

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