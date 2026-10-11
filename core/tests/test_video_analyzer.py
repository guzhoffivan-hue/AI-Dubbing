from app.core.media_info_store import MediaInfoStore
from app.services.video_analyzer import VideoAnalyzer
from app.services.analysis_service import AnalysisService


def test_video_analyzer(test_video):
    analyzer = VideoAnalyzer()

    info = analyzer.analyze(test_video)

    assert info.duration > 0

    assert info.video.width > 0
    assert info.video.height > 0
    assert info.video.fps > 0
    assert info.video.codec

    if info.audio_tracks:
        assert info.audio_tracks[0].codec
        assert info.audio_tracks[0].sample_rate > 0
        assert info.audio_tracks[0].channels > 0


def test_media_info_store(test_video, tmp_path):
    analyzer = VideoAnalyzer()
    store = MediaInfoStore()

    original = analyzer.analyze(test_video)

    path = store.save(original, tmp_path)

    assert path.exists()
    assert path == tmp_path / "analysis" / "media.json"

    loaded = store.load(tmp_path)

    assert loaded.duration == original.duration
    assert loaded.file_size == original.file_size

    assert loaded.video.width == original.video.width
    assert loaded.video.height == original.video.height
    assert loaded.video.fps == original.video.fps
    assert loaded.video.codec == original.video.codec

    assert len(loaded.audio_tracks) == len(
        original.audio_tracks
    )

    for loaded_track, original_track in zip(
        loaded.audio_tracks,
        original.audio_tracks,
    ):
        assert loaded_track.codec == original_track.codec
        assert (
            loaded_track.sample_rate
            == original_track.sample_rate
        )
        assert (
            loaded_track.channels
            == original_track.channels
        )
def test_analysis_service(test_video, tmp_path):
    service = AnalysisService()

    first = service.analyze(
        video_path=test_video,
        project_dir=tmp_path,
    )

    media_path = tmp_path / "analysis" / "media.json"

    assert media_path.exists()

    second = service.analyze(
        video_path=test_video,
        project_dir=tmp_path,
    )

    assert second.duration == first.duration
    assert second.file_size == first.file_size

    assert second.video.width == first.video.width
    assert second.video.height == first.video.height
    assert second.video.fps == first.video.fps
    assert second.video.codec == first.video.codec