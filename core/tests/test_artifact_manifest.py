import time
from pathlib import Path

from app.core.artifact_manifest import (
    ArtifactStore,
    build_manifest,
    fingerprint,
    verify_manifest,
)
from app.core.pipeline import Pipeline
from app.core.pipeline_state import PipelineState
from app.core.stage_handler import StageHandler, StageOutcome
from app.models.pipeline_stage import PipelineStage


class StubContext:
    def __init__(self, project_dir, config=None):
        self.project_dir = Path(project_dir)
        self.metadata = {}
        if config is not None:
            self.config = config


def test_fingerprint_rejects_missing_and_empty(tmp_path):
    assert fingerprint(None) is None
    assert fingerprint(tmp_path / "nope.bin") is None

    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert fingerprint(empty) is None

    real = tmp_path / "real.bin"
    real.write_bytes(b"x" * 10)
    assert fingerprint(real)["size"] == 10


def test_manifest_verifies_and_detects_change(tmp_path):
    artifact = tmp_path / "art.json"
    artifact.write_text("{}")
    source = tmp_path / "src.mp4"
    source.write_bytes(b"0" * 100)

    manifest = build_manifest(artifact, inputs=[source], parameters={"lang": "ru"})

    assert verify_manifest(manifest, inputs=[source], parameters={"lang": "ru"})[0] is True

    ok, reason = verify_manifest(manifest, inputs=[source], parameters={"lang": "en"})
    assert ok is False and "parameters" in reason

    source.write_bytes(b"1" * 250)
    ok, reason = verify_manifest(manifest, inputs=[source], parameters={"lang": "ru"})
    assert ok is False and "src.mp4" in reason


def test_manifest_detects_missing_artifact(tmp_path):
    artifact = tmp_path / "art.json"
    artifact.write_text("{}")

    manifest = build_manifest(artifact)
    assert verify_manifest(manifest)[0] is True

    artifact.unlink()
    assert verify_manifest(manifest)[0] is False


def test_manifest_refuses_unrecorded_input(tmp_path):
    artifact = tmp_path / "art.json"
    artifact.write_text("{}")
    other = tmp_path / "other.mp4"
    other.write_bytes(b"z" * 10)

    manifest = build_manifest(artifact)
    ok, reason = verify_manifest(manifest, inputs=[other])

    assert ok is False and "not recorded" in reason


def _run(project_dir, handler, context):
    pipeline = Pipeline(
        state=PipelineState.load(project_dir),
        project_dir=project_dir,
        handlers={PipelineStage.EXTRACT_AUDIO: handler},
    )
    pipeline.run(context)
    return pipeline


def test_completed_stage_is_rerun_when_input_changes(tmp_path):
    artifact = tmp_path / "audio.wav"
    artifact.write_bytes(b"a" * 64)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"0" * 100)

    calls = []

    def run(context):
        calls.append(1)
        artifact.write_bytes(b"a" * 64)
        return StageOutcome(artifact=artifact)

    handler = StageHandler(
        run=run,
        artifact=lambda context: artifact,
        inputs=lambda context: [source],
    )

    _run(tmp_path, handler, StubContext(tmp_path))
    assert len(calls) == 1

    _run(tmp_path, handler, StubContext(tmp_path))
    assert len(calls) == 1

    time.sleep(0.01)
    source.write_bytes(b"1" * 200)

    _run(tmp_path, handler, StubContext(tmp_path))
    assert len(calls) == 2

    manifest = ArtifactStore.load(tmp_path).get("extract_audio")
    assert manifest is not None
    assert manifest["inputs"]


def test_completed_stage_is_rerun_when_parameters_change(tmp_path):
    artifact = tmp_path / "audio.wav"
    artifact.write_bytes(b"a" * 64)

    calls = []

    def run(context):
        calls.append(1)
        artifact.write_bytes(b"a" * 64)
        return StageOutcome(artifact=artifact)

    handler = StageHandler(run=run, artifact=lambda context: artifact)

    _run(tmp_path, handler, StubContext(tmp_path, config={"lang": "ru"}))
    assert len(calls) == 1

    _run(tmp_path, handler, StubContext(tmp_path, config={"lang": "ru"}))
    assert len(calls) == 1

    _run(tmp_path, handler, StubContext(tmp_path, config={"lang": "en"}))
    assert len(calls) == 2


def test_stage_without_manifest_is_still_trusted(tmp_path):
    artifact = tmp_path / "audio.wav"
    artifact.write_bytes(b"a" * 64)

    calls = []

    def run(context):
        calls.append(1)
        return StageOutcome(artifact=artifact)

    handler = StageHandler(run=run, artifact=lambda context: artifact)

    _run(tmp_path, handler, StubContext(tmp_path))
    assert len(calls) == 1

    # Remove the manifest: a legacy project directory has none, and such a
    # saved completion keeps working exactly as before.
    (tmp_path / "artifact_manifest.json").unlink()

    _run(tmp_path, handler, StubContext(tmp_path))
    assert len(calls) == 1
