import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.models.stage_state import StageState, StageStatus


STATE_FILENAME = "pipeline_state.json"

# Bump when the persisted shape changes. A state file written by a NEWER version is
# quarantined instead of silently discarded, and a file that cannot be parsed is reported
# loudly: losing the record of a multi-hour run without a trace is unacceptable.
SCHEMA_VERSION = 2


def _keep_for_inspection(path: Path) -> None:
    """Keep a copy of an unreadable state file so the damage can be diagnosed.

    The original is left where it is (the next save would overwrite it, but the copy
    survives), so nothing about a long run disappears without a trace.
    """
    try:
        if not path.exists():
            return

        backup = path.with_name(
            f"{path.name}.damaged-{int(time.time())}"
        )
        backup.write_bytes(path.read_bytes())

        print(
            f"A copy of the damaged state was kept at {backup.name}",
            file=sys.stderr,
        )
    except OSError:
        pass


@dataclass
class PipelineState:
    stages: dict[str, StageState] = field(
        default_factory=dict
    )

    def get(self, stage: str) -> StageState:
        if stage not in self.stages:
            self.stages[stage] = StageState(
                stage=stage
            )

        return self.stages[stage]

    def is_completed(self, stage: str) -> bool:
        return (
            self.get(stage).status
            == StageStatus.COMPLETED
        )

    def save(self, project_dir: Path) -> Path:
        path = project_dir / STATE_FILENAME

        data = {
            # Version marker; load() skips keys starting with "_", and it doubles as the
            # signal that this file is not a bare legacy stage map.
            "_schema_version": SCHEMA_VERSION,
            **{
                stage: {
                    "stage": state.stage,
                    "status": state.status.value,
                    "progress": state.progress,
                    "message": state.message,
                }
                for stage, state in self.stages.items()
            },
        }

        # Atomic write: write to a temporary file, flush it to disk, then
        # replace the real file. A crash mid-write cannot corrupt the
        # previously saved state.
        temp_path = path.with_suffix(".json.tmp")

        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)

        return path

    @classmethod
    def load(cls, project_dir: Path) -> "PipelineState":
        path = project_dir / STATE_FILENAME

        if not path.exists():
            return cls()

        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )

            if not isinstance(data, dict):
                raise ValueError(
                    "state file is not a JSON object"
                )

            # Absent marker means the legacy flat stage map, which is still accepted.
            version = data.pop(
                "_schema_version", 1
            )

            if not isinstance(version, int):
                raise ValueError(
                    f"schema version is not an integer: {version!r}"
                )

            if version > SCHEMA_VERSION:
                quarantine = path.with_name(
                    f"{path.name}.unsupported-v{version}"
                )
                os.replace(path, quarantine)

                print(
                    "Pipeline state was written by a newer version "
                    f"(v{version} > v{SCHEMA_VERSION}). It was moved to "
                    f"{quarantine.name} and the run starts from an empty state; "
                    "completed stages will have to be recomputed.",
                    file=sys.stderr,
                )

                return cls()

            stages = {}

            for stage, value in data.items():
                if stage.startswith("_"):
                    continue

                if not isinstance(value, dict):
                    raise ValueError(
                        f"stage entry {stage!r} is not an object"
                    )

                stages[stage] = StageState(
                    stage=value["stage"],
                    status=StageStatus(value["status"]),
                    progress=value.get("progress", 0.0),
                    message=value.get("message", ""),
                )

            return cls(stages=stages)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ) as error:
            # A corrupt or partial state must not crash the application, but it must not
            # vanish quietly either: keep the file for inspection and say so loudly.
            print(
                f"WARNING: cannot read pipeline state {path} "
                f"({type(error).__name__}: {error}). Starting from an empty state; "
                "the damaged file was left in place for inspection.",
                file=sys.stderr,
            )

            _keep_for_inspection(path)

            return cls()
