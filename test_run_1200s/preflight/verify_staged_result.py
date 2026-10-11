#!/usr/bin/env python3
"""Verify the staged 1200-second test result (through segment_merge).

Checks the artifacts the staged run produced against the audio duration, and
compares the alignment timebase with what the pre-fix code would have produced.

Usage: verify_staged_result.py [work_dir]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TR = Path(__file__).resolve().parents[1]
WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else TR / 'work' / '42503442096_first1200s'
TOLERANCE = 2.0
EXPECTED_SECONDS = 1200.0
GAP_SECONDS = 30.0


def load(name):
    path = WORK / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        print(f'  !! {name} is not valid JSON: {exc}')
        return None


def ffprobe_duration(path):
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                   '-of', 'csv=p=0', str(path)], text=True).strip()
    return float(raw)


def main() -> int:
    print(f'work dir: {WORK}')
    problems, notes = [], []

    # --- audio -----------------------------------------------------------------
    audio = WORK / 'audio.wav'
    if audio.exists():
        duration = ffprobe_duration(audio)
        print(f'audio.wav        : {duration:.3f} s   ({audio.stat().st_size} bytes)')
        if abs(duration - EXPECTED_SECONDS) > 0.5:
            problems.append(f'audio.wav is {duration:.3f}s, expected {EXPECTED_SECONDS}s')
    else:
        problems.append('audio.wav missing')

    # --- chunks ---------------------------------------------------------------
    chunks = sorted((WORK / 'asr_chunks').glob('chunk_*.wav'))
    if chunks:
        durations = [ffprobe_duration(p) for p in chunks]
        total = sum(durations)
        print(f'asr_chunks       : {len(chunks)} files, total {total:.3f} s, '
              f'each {min(durations):.2f}..{max(durations):.2f} s')
        if abs(total - EXPECTED_SECONDS) > 0.5:
            problems.append(f'chunk durations sum to {total:.3f}s, expected {EXPECTED_SECONDS}s')
    else:
        problems.append('asr_chunks missing')

    # --- ASR ------------------------------------------------------------------
    asr = load('asr.json')
    if asr:
        text = asr.get('text', '')
        print(f'asr.json         : {len(text)} chars, language={asr.get("language")}, '
              f'segments={len(asr.get("segments", []))}')
        if not text.strip():
            problems.append('ASR transcript is empty')
    else:
        problems.append('asr.json missing')

    # --- alignment ------------------------------------------------------------
    alignment = load('alignment.json')
    if alignment:
        words = alignment.get('words') or []
        if not words:
            problems.append('alignment.json has no words')
        else:
            starts = sorted(float(w['start']) for w in words)
            ends = [float(w['end']) for w in words]
            max_end = max(ends)
            print(f'alignment.json   : {len(words)} words, {min(starts):.3f} .. {max_end:.3f} s')
            if max_end > EXPECTED_SECONDS + TOLERANCE:
                problems.append(f'alignment covers {max_end:.3f}s of a {EXPECTED_SECONDS}s audio')
            ordered = sorted(words, key=lambda w: float(w['start']))
            gaps = []
            reach = float(ordered[0]['end'])
            for word in ordered[1:]:
                start = float(word['start'])
                if start - reach > GAP_SECONDS:
                    gaps.append((reach, start, start - reach))
                reach = max(reach, float(word['end']))
            total_gap = sum(g[2] for g in gaps)
            print(f'  coverage gaps >{GAP_SECONDS:.0f}s : {len(gaps)}, total {total_gap:.1f} s')
            if gaps:
                problems.append(f'{len(gaps)} phantom coverage gaps totalling {total_gap:.1f}s')
            else:
                notes.append('no phantom coverage gaps (the 180s-offset bug produced 14 gaps / 968.8s '
                             'on the 1800s run)')
            legacy_end = (len(chunks) - 1) * 180.0 + max((ffprobe_duration(p) for p in chunks), default=120.0) \
                if chunks else None
            if legacy_end:
                print(f'  pre-fix code would have claimed up to {legacy_end:.1f} s '
                      f'(+{legacy_end - EXPECTED_SECONDS:.0f} s of phantom timeline)')
                notes.append(f'fix removed ~{legacy_end - EXPECTED_SECONDS:.0f}s of phantom timeline')
    else:
        problems.append('alignment.json missing')

    # --- diarization ----------------------------------------------------------
    diar = load('diarization.json')
    if diar:
        turns = diar.get('turns') or []
        if turns:
            end = max(float(t['end']) for t in turns)
            print(f'diarization.json : {len(turns)} turns, speakers={diar.get("speaker_count")}, '
                  f'ends {end:.3f} s')
            if end > EXPECTED_SECONDS + TOLERANCE:
                problems.append(f'diarization ends at {end:.3f}s beyond the audio')
        else:
            problems.append('diarization.json has no turns')
    else:
        problems.append('diarization.json missing')

    # --- merged segments ------------------------------------------------------
    segments = load('segments.json')
    if segments:
        lines = segments.get('lines') or []
        if lines:
            end = max(float(l['end']) for l in lines)
            start = min(float(l['start']) for l in lines)
            speakers = sorted({l.get('speaker') for l in lines})
            print(f'segments.json    : {len(lines)} lines, {start:.3f} .. {end:.3f} s, '
                  f'speakers={speakers}')
            if end > EXPECTED_SECONDS + TOLERANCE:
                problems.append(f'segments extend to {end:.3f}s beyond the audio')
            gaps = []
            for previous, current in zip(lines, lines[1:]):
                hole = float(current['start']) - float(previous['end'])
                if hole > 60.0:
                    gaps.append(hole)
            print(f'  line gaps >60s   : {len(gaps)}')
            if gaps:
                notes.append(f'{len(gaps)} long non-dialogue stretches in the source (>60s) — '
                             'legitimate if the source really is silent there')
            if not lines:
                problems.append('no merged segments')
        else:
            problems.append('segments.json has no lines')
    else:
        problems.append('segments.json missing')

    # --- status ---------------------------------------------------------------
    status = load('status.json')
    if status:
        stages = {k: v for k, v in status.items() if not k.startswith('_')}
        print('status.json      : ' + ', '.join(f'{k}={v}' for k, v in stages.items()))
        for stage in ('extract_audio', 'asr', 'alignment', 'diarization', 'segment_merge'):
            if stages.get(stage) != 'done':
                problems.append(f'stage {stage} is {stages.get(stage)!r}, expected done')
        if any(stages.get(s) == 'done' for s in ('translation', 'tts', 'mix', 'mux')):
            problems.append('a later stage ran although the run was supposed to stop after segment_merge')
    else:
        problems.append('status.json missing')

    # --- logs -----------------------------------------------------------------
    print('logs             :')
    for log in sorted((TR / 'logs').glob('*.log')):
        text = log.read_text(errors='replace')
        bad = [line for line in text.splitlines()
               if any(token in line for token in ('Traceback', 'RuntimeError', 'FileNotFoundError',
                                                  'out of memory', 'ERROR'))]
        print(f'  {log.name:44} {len(text.splitlines()):>5} lines, {len(bad)} error lines')
        for line in bad[:3]:
            print(f'      {line.strip()[:110]}')
        if bad:
            problems.append(f'{log.name} contains {len(bad)} error lines')

    print()
    if notes:
        print('notes:')
        for note in notes:
            print(f'  - {note}')
    print()
    if problems:
        print('RESULT: PROBLEMS FOUND')
        for problem in problems:
            print(f'  ! {problem}')
        return 1
    print('RESULT: all staged checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
