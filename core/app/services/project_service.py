from pathlib import Path
import shutil

from app.core.config_store import ConfigStore
from projects.project import Project
from projects.project_manager import ProjectManager


class ProjectService:
    def __init__(self, projects_dir: Path):
        self.manager = ProjectManager(projects_dir)

    def create_project(self, source_video: Path) -> Project:
        project = self.manager.create_project(source_video)

        source_dir = project.project_dir / "source"
        destination = source_dir / source_video.name

        shutil.copy2(source_video, destination)

        project.source_video = destination

        config_store = ConfigStore(project.project_dir)
        config_store.save(project.config)

        return project