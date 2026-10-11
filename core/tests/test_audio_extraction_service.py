from app.services.audio_extraction_service import (
    AudioExtractionService,
)


def test_audio_extraction_service(test_video, tmp_path):
    service = AudioExtractionService()

    first_result = service.extract_for_project(
        video_path=test_video,
        project_dir=tmp_path,
    )

    assert first_result.exists()
    assert first_result == (
        tmp_path / "audio" / "source_audio.wav"
    )
    assert first_result.stat().st_size > 0

    # Повторный вызов должен использовать существующий файл.
    original_modified_time = first_result.stat().st_mtime_ns

    second_result = service.extract_for_project(
        video_path=test_video,
        project_dir=tmp_path,
    )

    assert second_result == first_result
    assert second_result.stat().st_mtime_ns == original_modified_time
def test_audio_reextracted_when_source_changes(tmp_path, monkeypatch):
    service = AudioExtractionService()

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"first version")

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    output_path = audio_dir / "source_audio.wav"
    output_path.write_bytes(b"old audio")

    manifest_path = audio_dir / "source_audio.json"

    import json

    old_info = {
        "path": str(video_path.resolve()),
        "size": len(b"old version"),
        "modified_ns": video_path.stat().st_mtime_ns,
    }

    manifest_path.write_text(
        json.dumps(old_info),
        encoding="utf-8",
    )

    calls = []

    def fake_extract(video_path, output_path):
        calls.append(video_path)
        output_path.write_bytes(b"new audio")
        return output_path

    monkeypatch.setattr(
        service.extractor,
        "extract",
        fake_extract,
    )

    result = service.extract_for_project(
        video_path=video_path,
        project_dir=tmp_path,
    )

    assert len(calls) == 1
    assert result.read_bytes() == b"new audio"