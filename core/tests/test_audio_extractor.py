import json
import subprocess

from app.services.audio_extractor import AudioExtractor


def test_audio_extractor(test_video, tmp_path):
    output_path = tmp_path / "source_audio.wav"

    extractor = AudioExtractor()

    result = extractor.extract(
        video_path=test_video,
        output_path=output_path,
    )

    assert result.exists()
    assert result == output_path
    assert result.stat().st_size > 0
    assert result.suffix == ".wav"


def test_extracted_audio_format(test_video, tmp_path):
    output_path = tmp_path / "source_audio.wav"

    extractor = AudioExtractor()

    extractor.extract(
        video_path=test_video,
        output_path=output_path,
    )

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    info = json.loads(result.stdout)
    stream = info["streams"][0]

    assert stream["codec_name"] == "pcm_s16le"
    assert int(stream["sample_rate"]) == 16000
    assert int(stream["channels"]) == 1
    assert float(info["format"]["duration"]) > 0