import subprocess
from pathlib import Path


class AudioExtractor:
    def extract(
        self,
        video_path: Path,
        output_path: Path,
    ) -> Path:
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        if not output_path.exists():
            raise RuntimeError(
                "Audio extraction failed"
            )

        if output_path.stat().st_size == 0:
            raise RuntimeError(
                "Extracted audio file is empty"
            )

        return output_path