"""Cross-stage invariants: the checks that a stage must pass before it may run.

Artifact manifests answer "did this stage run on these inputs?". Invariants answer a
different question: "do the artifacts of the previous stages still agree with each other?".

The audit that started this work found TTS and mix consuming a translation of an older,
much shorter cut, because nothing verified that the translated lines matched the current
segments, or that the synthesized lines matched the dialogue timeline. A missing or stale
artifact must stop the pipeline loudly instead of producing a video whose speech ends in
the middle. These functions used to live only in ``scripts/integrity.py``; they are repeated
here so the core owns the rule and can enforce it before a stage starts.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

DURATION_TOLERANCE = 0.05

# Coverage thresholds, same values as scripts/integrity.py. A speech track that stops
# early, or leaves a multi-minute hole, is exactly the stale-translation failure mode.
MAX_LEADING_SILENCE_SECONDS = 120.0
MAX_TAIL_SILENCE_SECONDS = 60.0
MAX_INTERNAL_GAP_SECONDS = 60.0

# Log of invariant runs: [("translation", work_dir), ("tts", work_dir, audio_seconds)].
# Appended before the checks, so an attempted but failed validation is still visible.
CALLS: list[tuple] = []


class InvariantError(RuntimeError):
    """A cross-stage consistency rule was violated."""


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InvariantError(f"cannot read {path.name}: {error}") from error
    except ValueError as error:
        raise InvariantError(f"{path.name} is not valid JSON: {error}") from error


def _lines(path: Path) -> list[dict]:
    data = _read(path)

    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        raise InvariantError(f"{path.name} has no 'lines' list")

    for index, line in enumerate(data["lines"]):
        if not isinstance(line, dict):
            raise InvariantError(f"{path.name} line {index} is not an object")

    return data["lines"]


def _identity(line: dict) -> str:
    return str(line.get("line_id") or line.get("segment_id") or "")


def _start(line: dict) -> float:
    if "start" not in line:
        raise InvariantError(f"line {_identity(line) or '?'} has no start")
    return float(line["start"])


def wav_duration(path: Path) -> float:
    """Duration of a PCM wav file, using the standard library only."""

    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except (OSError, wave.Error) as error:
        raise InvariantError(f"cannot read audio {path.name}: {error}") from error

    if rate <= 0:
        raise InvariantError(f"audio {path.name} reports a zero sample rate")

    return frames / float(rate)


def validate_translation(work_dir: Path) -> None:
    """The translation must cover the current segments, one line each."""

    work_dir = Path(work_dir)
    CALLS.append(("translation", work_dir))
    segments_path = work_dir / "segments.json"
    translated_path = work_dir / "translated_segments.json"

    for path in (segments_path, translated_path):
        if not path.exists():
            raise InvariantError(f"{path.name} is missing in {work_dir}")

    segments = _lines(segments_path)
    translated = _lines(translated_path)

    if len(translated) != len(segments):
        raise InvariantError(
            "translation does not match the current segments: "
            f"{len(translated)} translated lines for {len(segments)} segments "
            "(a translation left over from an earlier cut?)"
        )

    for expected, actual in zip(segments, translated):
        left, right = _identity(expected), _identity(actual)

        if left != right:
            raise InvariantError(
                f"translation line order differs: expected {left}, found {right}"
            )

        if abs(_start(expected) - _start(actual)) > 1e-6:
            raise InvariantError(
                f"translation line {left} moved: segment starts at "
                f"{_start(expected)}, translation at {_start(actual)}"
            )

        if not str(actual.get("translated_text", "")).strip():
            raise InvariantError(f"translation line {left} is empty")

    if translated_path.stat().st_mtime_ns < segments_path.stat().st_mtime_ns:
        raise InvariantError(
            "translated_segments.json is older than segments.json; it was produced "
            "from a previous segmentation"
        )


def validate_tts_timeline(
    work_dir: Path,
    audio_seconds: float | None = None,
) -> None:
    """Synthesized lines must match the dialogue timeline and fit inside the audio."""

    work_dir = Path(work_dir)
    CALLS.append(("tts", work_dir, audio_seconds))
    dialogue_path = work_dir / "dialogue.json"
    manifest_path = work_dir / "tts_manifest.json"

    for path in (dialogue_path, manifest_path):
        if not path.exists():
            raise InvariantError(f"{path.name} is missing in {work_dir}")

    dialogue = _lines(dialogue_path)
    manifest = _lines(manifest_path)

    if len(manifest) != len(dialogue):
        raise InvariantError(
            "TTS manifest does not match the dialogue: "
            f"{len(manifest)} synthesized lines for {len(dialogue)} dialogue lines"
        )

    previous_start = None

    for expected, entry in zip(dialogue, manifest):
        left, right = _identity(expected), _identity(entry)

        if left != right:
            raise InvariantError(
                f"TTS manifest order differs: expected {left}, found {right}"
            )

        start = _start(entry)
        declared = float(entry.get("tts_duration", 0.0))

        if declared <= 0:
            raise InvariantError(f"synthesized line {left} declares no duration")

        path = Path(entry.get("path", ""))

        if not path.exists() or path.stat().st_size <= 0:
            raise InvariantError(
                f"synthesized line {left} has no audio file: {path}"
            )

        real = wav_duration(path)

        if abs(real - declared) > DURATION_TOLERANCE:
            raise InvariantError(
                f"synthesized line {left} declares {declared:.3f}s but the file "
                f"is {real:.3f}s"
            )

        if audio_seconds is not None and start + real > audio_seconds + DURATION_TOLERANCE:
            raise InvariantError(
                f"synthesized line {left} ends at {start + real:.2f}s, beyond the "
                f"{audio_seconds:.2f}s of audio"
            )

        if previous_start is not None and start < previous_start - 1e-6:
            raise InvariantError(
                f"synthesized line {left} starts at {start:.2f}s, before the "
                f"previous line at {previous_start:.2f}s"
            )

        previous_start = start

    # Coverage: the speech track must actually span the audio, not just be ordered.
    if manifest:
        leading = _start(manifest[0])
        if leading > MAX_LEADING_SILENCE_SECONDS:
            raise InvariantError(
                f"first synthesized line starts at {leading:.1f}s, more than "
                f"{MAX_LEADING_SILENCE_SECONDS:.0f}s of leading silence"
            )

        last = manifest[-1]
        last_end = _start(last) + wav_duration(Path(last.get("path", "")))

        if audio_seconds is not None:
            tail = audio_seconds - last_end
            if tail > MAX_TAIL_SILENCE_SECONDS:
                raise InvariantError(
                    f"synthesized audio ends at {last_end:.1f}s but the audio is "
                    f"{audio_seconds:.1f}s: {tail:.1f}s of trailing silence "
                    "(an incomplete TTS manifest?)"
                )

        for entry in manifest[1:]:
            gap = _start(entry) - previous_start
            if gap > MAX_INTERNAL_GAP_SECONDS:
                raise InvariantError(
                    f"synthesized line {_identity(entry)} leaves a {gap:.1f}s hole "
                    f"after {previous_start:.1f}s"
                )
            previous_start = _start(entry)
