from pathlib import Path

from app.core.media_info_store import MediaInfoStore
from app.models.media_info import MediaInfo
from app.services.video_analyzer import VideoAnalyzer


class AnalysisService:
    def __init__(self):
        self.analyzer = VideoAnalyzer()
        self.store = MediaInfoStore()

    def analyze(
        self,
        video_path: Path,
        project_dir: Path,
    ) -> MediaInfo:
        existing_path = (
            project_dir
            / "analysis"
            / "media.json"
        )

        if existing_path.exists():
            return self.store.load(project_dir)

        media_info = self.analyzer.analyze(video_path)

        self.store.save(
            media_info,
            project_dir,
        )

        return media_info