from dataclasses import dataclass


@dataclass
class VideoStreamInfo:
    width: int
    height: int
    fps: float
    codec: str


@dataclass
class AudioStreamInfo:
    codec: str
    sample_rate: int
    channels: int


@dataclass
class MediaInfo:
    duration: float
    file_size: int

    video: VideoStreamInfo
    audio_tracks: list[AudioStreamInfo]

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_tracks)

    @property
    def audio_track_count(self) -> int:
        return len(self.audio_tracks)