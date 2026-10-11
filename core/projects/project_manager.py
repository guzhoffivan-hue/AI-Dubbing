from pathlib import Path

from .project import Project


class ProjectManager:
    def __init__(self, projects_dir: Path):
        self.projects_dir = projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def create_project(self, source_video: Path) -> Project:
        project = Project.create(
            source_video=source_video,
            projects_dir=self.projects_dir,
        )

        project.project_dir.mkdir(parents=True, exist_ok=True)

        directories = [
            "source",
            "analysis",
            "transcription",
            "translation",
            "voices",
            "audio",
            "output",
        ]

        for directory in directories:
            (project.project_dir / directory).mkdir(exist_ok=True)

        return project