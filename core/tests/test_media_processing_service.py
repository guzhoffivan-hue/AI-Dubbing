from pathlib import Path

from app.models.media_info import (
    AudioStreamInfo,
    MediaInfo,
    VideoStreamInfo,
)
from app.services.analysis_service import AnalysisService
from app.services.audio_extraction_service import (
    AudioExtractionService,
)
from app.services.media_processing_service import (
    MediaProcessingService,
)


def build_media_info() -> MediaInfo:
    return MediaInfo(
        duration=12.5,
        file_size=1024,
        video=VideoStreamInfo(
            width=1920,
            height=1080,
            fps=30.0,
            codec="h264",
        ),
        audio_tracks=[
            AudioStreamInfo(
                codec="aac",
                sample_rate=48000,
                channels=2,
            ),
        ],
    )


def test_service_builds_dependencies():
    service = MediaProcessingService()

    assert isinstance(
        service.analysis_service,
        AnalysisService,
    )
    assert isinstance(
        service.audio_extraction_service,
        AudioExtractionService,
    )


def test_process_returns_media_info_and_audio_path(
    tmp_path,
    monkeypatch,
):
    service = MediaProcessingService()

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    expected_media_info = build_media_info()
    expected_audio_path = (
        tmp_path / "audio" / "source_audio.wav"
    )

    calls = []

    def fake_analyze(video_path, project_dir):
        calls.append(
            ("analyze", video_path, project_dir)
        )
        return expected_media_info

    def fake_extract(video_path, project_dir):
        calls.append(
            ("extract", video_path, project_dir)
        )
        return expected_audio_path

    monkeypatch.setattr(
        service.analysis_service,
        "analyze",
        fake_analyze,
    )
    monkeypatch.setattr(
        service.audio_extraction_service,
        "extract_for_project",
        fake_extract,
    )

    media_info, audio_path = service.process(
        video_path=video_path,
        project_dir=tmp_path,
    )

    assert media_info is expected_media_info
    assert audio_path == expected_audio_path

    assert calls == [
        ("analyze", video_path, tmp_path),
        ("extract", video_path, tmp_path),
    ]


def test_process_delegates_with_given_paths(
    tmp_path,
    monkeypatch,
):
    service = MediaProcessingService()

    video_path = tmp_path / "input" / "video.mp4"
    project_dir = tmp_path / "project"

    received = {}

    def fake_analyze(video_path, project_dir):
        received["analyze_video"] = video_path
        received["analyze_project"] = project_dir
        return build_media_info()

    def fake_extract(video_path, project_dir):
        received["extract_video"] = video_path
        received["extract_project"] = project_dir
        return Path("audio.wav")

    monkeypatch.setattr(
        service.analysis_service,
        "analyze",
        fake_analyze,
    )
    monkeypatch.setattr(
        service.audio_extraction_service,
        "extract_for_project",
        fake_extract,
    )

    service.process(
        video_path=video_path,
        project_dir=project_dir,
    )

    assert received["analyze_video"] == video_path
    assert received["analyze_project"] == project_dir
    assert received["extract_video"] == video_path
    assert received["extract_project"] == project_dir
