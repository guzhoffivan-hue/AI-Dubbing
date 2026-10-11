#!/usr/bin/env python3
"""Cross-artifact integrity checks for the dubbing pipeline.

These checks exist because the pipeline once shipped a 30-minute video whose Russian
speech stopped after 3.5 minutes: a stale 43-segment translation from an earlier
210-second run was accepted as valid input for an 1800-second run, TTS happily
synthesized those 43 lines, and the mixer padded the remaining 1590 seconds with
silence. Nothing in the pipeline noticed, because every stage only checked that its
own output file existed.

Policy: every check fails loudly. A stage must never turn incomplete input into an
externally successful artifact.

Only the standard library is used here, so the checks stay unit-testable without
loading models or installing dependencies.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

# --- speech coverage policy -------------------------------------------------
# Real material may legitimately start or end with music, so leading and trailing
# non-speech are tolerated up to these bounds. Anything larger means a whole stretch
# of the video has no dubbing at all, which must be an error rather than a result.
MAX_LEADING_SILENCE_SECONDS = 120.0
MAX_TAIL_SILENCE_SECONDS = 60.0
MAX_INTERNAL_GAP_SECONDS = 60.0
# Float timestamps travel through JSON, so compare with a small tolerance.
TIME_TOLERANCE_SECONDS = 0.001
# A manifest line's declared duration must match the audio file on disk.
DURATION_TOLERANCE_SECONDS = 0.05
# Stop after this many problems; the full list is not useful for a human reader.
MAX_REPORTED_PROBLEMS = 10


class IntegrityError(RuntimeError):
    """Raised when pipeline artifacts are missing, stale, partial, or inconsistent."""


def _raise(headline, problems):
    detail = '\n'.join('  - ' + problem for problem in problems[:MAX_REPORTED_PROBLEMS])
    extra = '' if len(problems) <= MAX_REPORTED_PROBLEMS else f'\n  ... and {len(problems) - MAX_REPORTED_PROBLEMS} more'
    raise IntegrityError(f'{headline}\n{detail}{extra}')


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise IntegrityError(f'Required artifact is missing: {path}')
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise IntegrityError(f'Cannot read artifact {path}: {type(exc).__name__}: {exc}') from exc


def audio_duration(path):
    """Duration in seconds of a WAV file; falls back to ffprobe for other formats."""
    path = Path(path)
    if not path.exists():
        raise IntegrityError(f'Audio file is missing: {path}')
    if path.stat().st_size == 0:
        raise IntegrityError(f'Audio file is empty: {path}')
    try:
        with wave.open(str(path)) as handle:
            rate = handle.getframerate()
            if rate <= 0:
                raise IntegrityError(f'Audio file has an invalid sample rate: {path}')
            return handle.getnframes() / float(rate)
    except wave.Error:
        pass
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        raise IntegrityError(f'Cannot read {path} as WAV and ffprobe is unavailable')
    try:
        raw = subprocess.check_output(
            [ffprobe, '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
            text=True).strip()
        return float(raw)
    except Exception as exc:
        raise IntegrityError(f'Cannot determine duration of {path}: {type(exc).__name__}: {exc}') from exc


def _same_time(left, right):
    return abs(float(left) - float(right)) <= TIME_TOLERANCE_SECONDS


def _id_of(line):
    return line.get('segment_id') or line.get('line_id')


def speech_intervals(lines):
    """Speech intervals in playback order, using each line's synthesized duration."""
    return sorted((float(line['start']), float(line['start']) + float(line['tts_duration'])) for line in lines)


def coverage_report(intervals, audio_seconds):
    """(leading, internal_gaps, tail) seconds of the audio not covered by speech."""
    leading = max(0.0, intervals[0][0])
    tail = max(0.0, float(audio_seconds) - max(end for _, end in intervals))
    gaps = []
    reach = intervals[0][1]
    for start, end in intervals[1:]:
        if start - reach > MAX_INTERNAL_GAP_SECONDS:
            gaps.append((round(reach, 3), round(start, 3), round(start - reach, 3)))
        reach = max(reach, end)
    return leading, gaps, tail


def validate_translation(work, segments_name='segments.json', translated_name='translated_segments.json'):
    """Prove that the translation covers the CURRENT segment set, completely.

    Rejects a partially written file (translate.py saves every 50 lines, so a killed
    run leaves a plausible-looking short file) and a leftover translation from an
    earlier run whose segment set was different.
    """
    work = Path(work)
    segments_path = work / segments_name
    translated_path = work / translated_name
    segments = load_json(segments_path)
    translated = load_json(translated_path)
    segment_lines = segments.get('lines') or []
    translated_lines = translated.get('lines') or []
    if not segment_lines:
        raise IntegrityError(f'{segments_path} contains no segments; run segment_merge first')

    problems = []
    if len(translated_lines) != len(segment_lines):
        problems.append(f'covers {len(translated_lines)} of {len(segment_lines)} segments')
    if translated_path.stat().st_mtime_ns < segments_path.stat().st_mtime_ns:
        problems.append(
            f'{translated_name} is older than {segments_name}; it describes an earlier segment set')
    for index, (segment, line) in enumerate(zip(segment_lines, translated_lines)):
        expected_id = _id_of(segment)
        if expected_id != _id_of(line):
            problems.append(
                f'segment {index}: id mismatch ({segments_name}={expected_id!r}, {translated_name}={_id_of(line)!r})')
        elif not _same_time(segment['start'], line['start']) or not _same_time(segment['end'], line['end']):
            problems.append(
                f'segment {expected_id}: time mismatch ({segments_name}={segment["start"]}-{segment["end"]}, '
                f'{translated_name}={line["start"]}-{line["end"]})')
        if not str(line.get('translated_text', '')).strip():
            problems.append(f'segment {_id_of(line)}: translated_text is empty')
        if len(problems) >= MAX_REPORTED_PROBLEMS:
            break
    if problems:
        _raise(f'Translation does not match {segments_name} and cannot be dubbed', problems)
    return segment_lines, translated_lines


def validate_tts_timeline(work, dialogue_name='dialogue.json', manifest_name='tts_manifest.json',
                          meta_name='audio_meta.json'):
    """Prove that the TTS manifest belongs to the current dialogue and covers the audio.

    Detects a stale manifest (produced for a different segment set), missing or empty
    audio files, durations that disagree with the files on disk, out-of-range timings,
    unexplained internal gaps, and a truncated speech track.
    """
    work = Path(work)
    dialogue_path = work / dialogue_name
    manifest_path = work / manifest_name
    dialogue = load_json(dialogue_path)
    manifest = load_json(manifest_path)
    meta = load_json(work / meta_name)
    if 'duration' not in meta:
        raise IntegrityError(f'{work / meta_name} has no duration field')
    audio_seconds = float(meta['duration'])
    dialogue_lines = dialogue.get('lines') or []
    manifest_lines = manifest.get('lines') or []
    if not dialogue_lines:
        raise IntegrityError(f'{dialogue_path} contains no dialogue lines')
    if not manifest_lines:
        raise IntegrityError(f'{manifest_path} contains no TTS lines')

    problems = []
    if len(manifest_lines) != len(dialogue_lines):
        problems.append(
            f'TTS manifest has {len(manifest_lines)} lines for {len(dialogue_lines)} dialogue lines')

    if not problems:
        for index, (dialogue_line, manifest_line) in enumerate(zip(dialogue_lines, manifest_lines)):
            expected_id = _id_of(dialogue_line)
            if expected_id != _id_of(manifest_line):
                problems.append(
                    f'line {index}: manifest is for {_id_of(manifest_line)!r} but the dialogue line is {expected_id!r}'
                    ' (stale manifest from an earlier segment set)')
            elif not _same_time(dialogue_line['start'], manifest_line.get('start', -1)):
                problems.append(
                    f'line {expected_id}: manifest start {manifest_line.get("start")} != dialogue start {dialogue_line["start"]}')
            elif str(dialogue_line.get('translated_text', '')) != str(manifest_line.get('translated_text', '')):
                problems.append(f'line {expected_id}: manifest text does not match the dialogue text')
            if len(problems) >= MAX_REPORTED_PROBLEMS:
                break

    if not problems:
        for index, line in enumerate(manifest_lines):
            line_id = _id_of(line) or f'#{index}'
            path = line.get('path')
            if not path:
                problems.append(f'line {line_id}: no audio path in the manifest')
                continue
            if not Path(path).exists():
                problems.append(f'line {line_id}: audio file is missing ({path})')
                continue
            if Path(path).stat().st_size == 0:
                problems.append(f'line {line_id}: audio file is empty ({path})')
                continue
            declared = float(line.get('tts_duration', 0.0))
            if declared <= 0:
                problems.append(f'line {line_id}: declared tts_duration is {declared}')
                continue
            try:
                real = audio_duration(path)
            except IntegrityError as exc:
                problems.append(f'line {line_id}: {exc}')
                continue
            if abs(real - declared) > DURATION_TOLERANCE_SECONDS:
                problems.append(
                    f'line {line_id}: declared tts_duration {declared:.3f}s but the file is {real:.3f}s ({path})')
            start = float(line.get('start', -1))
            if start < 0:
                problems.append(f'line {line_id}: negative start {start}')
            elif start + declared > audio_seconds + DURATION_TOLERANCE_SECONDS:
                problems.append(
                    f'line {line_id}: ends at {start + declared:.3f}s, beyond the {audio_seconds:.3f}s audio')
            if len(problems) >= MAX_REPORTED_PROBLEMS:
                break

    if not problems:
        starts = [float(line['start']) for line in manifest_lines]
        for index in range(1, len(starts)):
            if starts[index] < starts[index - 1] - TIME_TOLERANCE_SECONDS:
                problems.append(
                    f'line {_id_of(manifest_lines[index])}: starts at {starts[index]:.3f}s, '
                    f'before the previous line at {starts[index - 1]:.3f}s')
                break

    if not problems:
        intervals = speech_intervals(manifest_lines)
        leading, gaps, tail = coverage_report(intervals, audio_seconds)
        speech_end = max(end for _, end in intervals)
        if tail > MAX_TAIL_SILENCE_SECONDS:
            problems.append(
                f'speech ends at {speech_end:.3f}s but the audio is {audio_seconds:.3f}s long: '
                f'{tail:.3f}s ({tail / audio_seconds * 100:.1f}%) has no dubbing at all')
        if leading > MAX_LEADING_SILENCE_SECONDS:
            problems.append(f'the first {leading:.3f}s of the audio has no dubbing')
        for gap_start, gap_end, gap in gaps:
            problems.append(f'no dubbing between {gap_start:.3f}s and {gap_end:.3f}s ({gap:.3f}s hole)')

    if problems:
        _raise(f'TTS timeline is incomplete for {work} and must not be mixed', problems)
    return manifest_lines
