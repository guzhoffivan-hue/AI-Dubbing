from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from app.models.project_config import ProjectConfig


@dataclass
class Project:
    id: UUID
    name: str
    source_video: Path
    project_dir: Path
    config: ProjectConfig

    @classmethod
    def create(cls, source_video: Path, projects_dir: Path) -> "Project":
        project_id = uuid4()
        name = source_video.stem

        project_dir = projects_dir / str(project_id)

        return cls(
            id=project_id,
            name=name,
            source_video=source_video,
            project_dir=project_dir,
            config=ProjectConfig.default(),
        )