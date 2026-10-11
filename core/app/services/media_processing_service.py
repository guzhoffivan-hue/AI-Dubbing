from pathlib import Path

from app.models.media_info import MediaInfo
from app.services.analysis_service import AnalysisService
from app.services.audio_extraction_service import (
    AudioExtractionService,
)


class MediaProcessingService:
    def __init__(self):
        self.analysis_service = AnalysisService()
        self.audio_extraction_service = AudioExtractionService()

    def process(
        self,
        video_path: Path,
        project_dir: Path,
    ) -> tuple[MediaInfo, Path]:
        media_info = self.analysis_service.analyze(
            video_path=video_path,
            project_dir=project_dir,
        )

        audio_path = self.audio_extraction_service.extract_for_project(
            video_path=video_path,
            project_dir=project_dir,
        )

        return media_info, audio_path