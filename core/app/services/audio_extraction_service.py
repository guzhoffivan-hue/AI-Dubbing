import json
from pathlib import Path

from app.services.audio_extractor import AudioExtractor


class AudioExtractionService:
    def __init__(self):
        self.extractor = AudioExtractor()

    def extract_for_project(
        self,
        video_path: Path,
        project_dir: Path,
    ) -> Path:
        video_path = video_path.resolve()

        audio_dir = project_dir / "audio"
        audio_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = audio_dir / "source_audio.wav"
        manifest_path = audio_dir / "source_audio.json"

        source_info = {
            "path": str(video_path),
            "size": video_path.stat().st_size,
            "modified_ns": video_path.stat().st_mtime_ns,
        }

        if (
            output_path.exists()
            and output_path.stat().st_size > 0
            and manifest_path.exists()
        ):
            try:
                saved_info = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )

                if saved_info == source_info:
                    return output_path
            except (OSError, json.JSONDecodeError):
                pass

        self.extractor.extract(
            video_path=video_path,
            output_path=output_path,
        )

        manifest_path.write_text(
            json.dumps(
                source_info,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return output_path