import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_video(tmp_path_factory) -> Path:
    """Generate a tiny self-contained video (video + audio) with FFmpeg.

    Tests must not depend on a user file on an external disk. The clip is
    created once per test session; the test is skipped only when FFmpeg /
    ffprobe are genuinely unavailable.
    """

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe are not installed")

    directory = tmp_path_factory.mktemp("media")
    video_path = directory / "sample.mp4"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=25",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100",
        "-t",
        "2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(
            "could not generate test video with ffmpeg: "
            f"{result.stderr.strip()[:300]}"
        )

    return video_path
