import json
from dataclasses import asdict
from pathlib import Path

from app.models.project_config import ProjectConfig


class ConfigStore:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.config_path = project_dir / "config.json"

    def save(self, config: ProjectConfig) -> None:
        data = asdict(config)

        with self.config_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(self) -> ProjectConfig:
        if not self.config_path.exists():
            return ProjectConfig.default()

        with self.config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return ProjectConfig(
            translation=ProjectConfig.default().translation.__class__(
                **data["translation"]
            ),
            transcription=ProjectConfig.default().transcription.__class__(
                **data["transcription"]
            ),
            dubbing=ProjectConfig.default().dubbing.__class__(
                **data["dubbing"]
            ),
        )