"""Artifact manifests: what makes a saved stage completion trustworthy.

``exists() and size > 0`` is not proof of correctness. A file left over from an earlier
run, or one produced from different inputs, passes that test — and that is exactly how a
stale translation of an old 43-segment cut was consumed by TTS and mix while the audio was
thirty minutes long.

A manifest records three things about a completed stage: the identity of its artifact
(absolute path, size, mtime in nanoseconds), the fingerprints of the inputs it declared,
and a hash of the parameters that shaped the run. A saved completion may be skipped only
while all three still match.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

MANIFEST_FILENAME = "artifact_manifest.json"
MANIFEST_VERSION = 1


def fingerprint(path: Path | str | None) -> dict | None:
    """Identity of a file: absolute path, size and mtime in nanoseconds.

    Returns ``None`` for a missing or empty file, so an empty file can never be
    mistaken for a valid artifact.
    """

    if path is None:
        return None

    candidate = Path(path)

    try:
        stat = candidate.stat()
    except OSError:
        return None

    if stat.st_size <= 0:
        return None

    return {
        "path": str(candidate.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def fingerprint_matches(record: object) -> bool:
    """True when the file recorded in ``record`` is still byte-identical in size/mtime."""

    if not isinstance(record, dict):
        return False

    current = fingerprint(record.get("path"))

    if current is None:
        return False

    return (
        current["size"] == record.get("size")
        and current["mtime_ns"] == record.get("mtime_ns")
    )


def parameters_hash(parameters: object) -> str:
    """Stable hash of the run parameters (language, models, voices, ...)."""

    if not parameters:
        return ""

    try:
        blob = json.dumps(
            parameters,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError):
        blob = str(parameters)

    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_manifest(
    artifact: Path | str | None,
    inputs: list[Path | str] | None = None,
    parameters: object = None,
) -> dict | None:
    """Describe a finished stage, or ``None`` when it produced no usable artifact."""

    recorded = fingerprint(artifact)

    if recorded is None:
        return None

    return {
        "version": MANIFEST_VERSION,
        "artifact": recorded,
        "inputs": [
            item
            for item in (
                fingerprint(candidate)
                for candidate in (inputs or [])
            )
            if item is not None
        ],
        "parameters_sha256": parameters_hash(parameters),
        "recorded_at": int(time.time()),
    }


def verify_manifest(
    manifest: object,
    inputs: list[Path | str] | None = None,
    parameters: object = None,
) -> tuple[bool, str]:
    """Can a stage with this manifest still be skipped? Returns (ok, reason)."""

    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != MANIFEST_VERSION
    ):
        return False, "no usable manifest"

    if not fingerprint_matches(manifest.get("artifact")):
        return False, "artifact changed or is missing"

    if parameters_hash(parameters) != manifest.get(
        "parameters_sha256",
        "",
    ):
        return False, "run parameters changed"

    recorded = {
        item["path"]: item
        for item in manifest.get("inputs") or []
        if isinstance(item, dict) and "path" in item
    }

    for candidate in inputs or []:
        key = str(Path(candidate).resolve())
        entry = recorded.get(key)

        if entry is None:
            return False, f"input {Path(candidate).name} was not recorded"

        if not fingerprint_matches(entry):
            return False, f"input {Path(candidate).name} changed"

    return True, "manifest matches"


class ArtifactStore:
    """Per-project map of stage name -> manifest, written atomically."""

    def __init__(
        self,
        project_dir: Path | str | None = None,
        manifests: dict | None = None,
    ):
        self.project_dir = (
            Path(project_dir)
            if project_dir is not None
            else None
        )
        self.manifests: dict[str, dict] = dict(
            manifests or {}
        )

    @property
    def path(self) -> Path | None:
        if self.project_dir is None:
            return None

        return self.project_dir / MANIFEST_FILENAME

    def get(self, stage: object) -> dict | None:
        return self.manifests.get(str(stage))

    def record(
        self,
        stage: object,
        manifest: dict | None,
    ) -> None:
        if manifest is not None:
            self.manifests[str(stage)] = manifest

    def save(self) -> Path | None:
        path = self.path

        if path is None:
            return None

        temp_path = path.with_name(
            path.name + ".tmp"
        )

        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "version": MANIFEST_VERSION,
                    "stages": self.manifests,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)

        return path

    @classmethod
    def load(
        cls,
        project_dir: Path | str | None,
    ) -> "ArtifactStore":
        if project_dir is None:
            return cls()

        path = Path(project_dir) / MANIFEST_FILENAME

        if not path.exists():
            return cls(project_dir)

        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            # A damaged manifest must never be trusted, but it also must not
            # crash the run: an empty store simply means nothing is skippable.
            return cls(project_dir)

        stages = (
            data.get("stages")
            if isinstance(data, dict)
            else None
        )

        if not isinstance(stages, dict):
            return cls(project_dir)

        return cls(project_dir, stages)
