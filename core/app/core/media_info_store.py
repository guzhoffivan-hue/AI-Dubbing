import json
from dataclasses import asdict
from pathlib import Path

from app.models.media_info import (
    AudioStreamInfo,
    MediaInfo,
    VideoStreamInfo,
)


class MediaInfoStore:
    def save(self, media_info: MediaInfo, project_dir: Path) -> Path:
        analysis_dir = project_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        path = analysis_dir / "media.json"

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                asdict(media_info),
                file,
                ensure_ascii=False,
                indent=2,
            )

        return path

    def load(self, project_dir: Path) -> MediaInfo:
        path = project_dir / "analysis" / "media.json"

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return MediaInfo(
            duration=data["duration"],
            file_size=data["file_size"],
            video=VideoStreamInfo(
                width=data["video"]["width"],
                height=data["video"]["height"],
                fps=data["video"]["fps"],
                codec=data["video"]["codec"],
            ),
            audio_tracks=[
                AudioStreamInfo(
                    codec=track["codec"],
                    sample_rate=track["sample_rate"],
                    channels=track["channels"],
                )
                for track in data["audio_tracks"]
            ],
        )