import json
import os
import wave
from pathlib import Path

import pytest

from app.core.pipeline import Pipeline
from app.core.pipeline_state import PipelineState
from app.core.stage_handler import StageHandler
from app.invariants import (
    InvariantError,
    validate_translation,
    validate_tts_timeline,
)
from app.models.pipeline_stage import PipelineStage


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_wav(path, seconds, rate=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))


def make_translation(work, translated_text="Привет", stale=False):
    write_json(
        work / "segments.json",
        {"lines": [{"line_id": "LINE_0001", "start": 0.0, "end": 2.0, "text": "你好"}]},
    )
    write_json(
        work / "translated_segments.json",
        {"lines": [{"line_id": "LINE_0001", "start": 0.0, "translated_text": translated_text}]},
    )

    if stale:
        older = (work / "segments.json").stat().st_mtime_ns - 10**9
        os.utime(work / "translated_segments.json", ns=(older, older))


def make_timeline(work, declared=None):
    write_json(work / "dialogue.json", {"lines": [{"segment_id": "LINE_0001", "start": 0.0}]})
    wav = work / "tts" / "0000.wav"
    write_wav(wav, 1.0)
    write_json(
        work / "tts_manifest.json",
        {
            "lines": [
                {
                    "segment_id": "LINE_0001",
                    "start": 0.0,
                    "tts_duration": 1.0 if declared is None else declared,
                    "path": str(wav),
                }
            ]
        },
    )


def test_translation_must_cover_the_current_segments(tmp_path):
    make_translation(tmp_path)
    validate_translation(tmp_path)

    write_json(tmp_path / "translated_segments.json", {"lines": []})

    with pytest.raises(InvariantError, match="does not match"):
        validate_translation(tmp_path)


def test_translation_rejects_empty_and_stale_lines(tmp_path):
    make_translation(tmp_path, translated_text="   ")

    with pytest.raises(InvariantError, match="empty"):
        validate_translation(tmp_path)

    make_translation(tmp_path, stale=True)

    with pytest.raises(InvariantError, match="older than"):
        validate_translation(tmp_path)


def test_tts_timeline_checks_duration_file_and_bounds(tmp_path):
    make_timeline(tmp_path)
    # 1.5 s of audio: the single 1 s line covers it (tail < MAX_TAIL_SILENCE)
    validate_tts_timeline(tmp_path, audio_seconds=1.5)

    make_timeline(tmp_path, declared=5.0)
    with pytest.raises(InvariantError, match="declares"):
        validate_tts_timeline(tmp_path)

    make_timeline(tmp_path)
    (tmp_path / "tts" / "0000.wav").unlink()
    with pytest.raises(InvariantError, match="no audio file"):
        validate_tts_timeline(tmp_path)

    make_timeline(tmp_path)
    with pytest.raises(InvariantError, match="beyond"):
        validate_tts_timeline(tmp_path, audio_seconds=0.5)


def test_tts_timeline_rejects_a_short_manifest(tmp_path):
    make_timeline(tmp_path)
    write_json(tmp_path / "dialogue.json", {"lines": [{"segment_id": "A", "start": 0.0}, {"segment_id": "B", "start": 3.0}]})

    with pytest.raises(InvariantError, match="does not match the dialogue"):
        validate_tts_timeline(tmp_path)


def test_precondition_stops_a_stage_before_it_runs(tmp_path):
    ran = []

    def precondition(context):
        raise InvariantError("refusing to run on a stale artifact")

    handler = StageHandler(run=lambda context: ran.append(1), precondition=precondition)
    pipeline = Pipeline(
        state=PipelineState(),
        project_dir=tmp_path,
        handlers={PipelineStage.EXTRACT_AUDIO: handler},
    )
    pipeline.run(None)

    assert ran == []

    state = PipelineState.load(tmp_path).get("extract_audio")
    assert state.status.value == "failed"
    assert "stale artifact" in state.message


def test_stage_without_precondition_still_runs(tmp_path):
    ran = []

    handler = StageHandler(run=lambda context: ran.append(1))
    pipeline = Pipeline(
        state=PipelineState(),
        project_dir=tmp_path,
        handlers={PipelineStage.EXTRACT_AUDIO: handler},
    )
    pipeline.run(None)

    assert ran == [1]
    assert PipelineState.load(tmp_path).get("extract_audio").status.value == "completed"
