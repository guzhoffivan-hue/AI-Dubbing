import json
import subprocess
from pathlib import Path

from app.models.media_info import (
    AudioStreamInfo,
    MediaInfo,
    VideoStreamInfo,
)


class VideoAnalyzer:
    def analyze(self, video_path: Path) -> MediaInfo:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        data = json.loads(result.stdout)

        streams = data.get("streams", [])
        format_data = data.get("format", {})

        video_stream = self._find_video_stream(streams)
        audio_streams = self._find_audio_streams(streams)

        return MediaInfo(
            duration=float(format_data.get("duration", 0)),
            file_size=int(format_data.get("size", 0)),
            video=VideoStreamInfo(
                width=int(video_stream.get("width", 0)),
                height=int(video_stream.get("height", 0)),
                fps=self._parse_fps(
                    video_stream.get("r_frame_rate", "0/1")
                ),
                codec=video_stream.get("codec_name", "unknown"),
            ),
            audio_tracks=[
                AudioStreamInfo(
                    codec=stream.get("codec_name", "unknown"),
                    sample_rate=int(
                        stream.get("sample_rate", 0)
                    ),
                    channels=int(
                        stream.get("channels", 0)
                    ),
                )
                for stream in audio_streams
            ],
        )

    @staticmethod
    def _find_video_stream(streams: list[dict]) -> dict:
        for stream in streams:
            if stream.get("codec_type") == "video":
                return stream

        raise ValueError("Video stream not found")

    @staticmethod
    def _find_audio_streams(streams: list[dict]) -> list[dict]:
        return [
            stream
            for stream in streams
            if stream.get("codec_type") == "audio"
        ]

    @staticmethod
    def _parse_fps(value: str) -> float:
        numerator, denominator = value.split("/")

        denominator = float(denominator)

        if denominator == 0:
            return 0.0

        return float(numerator) / denominator