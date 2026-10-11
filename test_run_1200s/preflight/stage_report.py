#!/usr/bin/env python3
"""Produce the 10-point stage report for the 1200-second test fragment.

Reads only the artifacts the run itself produced; writes nothing.

Usage: stage_report.py [work_dir]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

TR = Path(__file__).resolve().parents[1]
WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else TR / 'work' / '42503442096_first1200s'
LOGS = TR / 'logs'
AUDIO_TOL = 0.5
GAP_ALIGN = 30.0
GAP_LINE = 60.0


def load(name):
    path = WORK / name
    return json.loads(path.read_text()) if path.exists() else None


def dur(path):
    return float(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
        text=True).strip())


def mtime(name):
    path = WORK / name
    if not path.exists():
        return 'missing'
    return f"{__import__('datetime').datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M:%S}"


def head(n, title):
    print(f'\n{"=" * 78}\n{n}. {title}\n{"=" * 78}')


def main() -> int:
    status = load('status.json') or {}
    signature = status.get('_signature', {})

    head(10, 'ARTIFACT PROVENANCE (reported first because everything else depends on it)')
    print(f'  work dir                    : {WORK}')
    print(f'  input video (signature)     : {signature.get("path")}')
    print(f'  signature size / mtime_ns   : {signature.get("size")} / {signature.get("mtime_ns")}')
    fragment = Path(signature.get('path', ''))
    if fragment.exists():
        st = fragment.stat()
        print(f'  fragment on disk now        : size {st.st_size} / mtime_ns {st.st_mtime_ns}  '
              f'-> {"MATCH" if (st.st_size == signature.get("size") and st.st_mtime_ns == signature.get("mtime_ns")) else "MISMATCH"}')
        print(f'  fragment md5                : {subprocess.check_output(["md5", "-q", str(fragment)], text=True).strip()}')
    print(f'  signature mode/start/dur    : {signature.get("mode")} / {signature.get("start")} / {signature.get("duration")}')
    asr_doc = load('asr.json') or {}
    print(f'  asr.json "audio" field      : {asr_doc.get("audio")}')
    print(f'  asr.json model              : {asr_doc.get("model")}')
    print('  artifact mtimes (all must fall inside the run window):')
    for name in ('audio.wav', 'audio_meta.json', 'asr.json', 'alignment.json', 'diarization.json',
                 'speakers.json', 'segments.json', 'status.json'):
        print(f'    {name:18} {mtime(name)}')
    print('  stage status:')
    for key, value in status.items():
        if not key.startswith('_'):
            print(f'    {key:15} {value}')

    head(1, 'ACTUAL AUDIO DURATION')
    audio = WORK / 'audio.wav'
    audio_seconds = dur(audio) if audio.exists() else 0.0
    probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                     'stream=codec_name,sample_rate,channels:format=duration,size',
                                     '-of', 'json', str(audio)], text=True)
    info = json.loads(probe)
    print(f'  path      : {audio}')
    print(f'  duration  : {audio_seconds:.3f} s   (extraction request: 1200.0 s)')
    print(f'  size      : {info["format"]["size"]} bytes')
    for stream in info.get('streams', []):
        print(f'  stream    : {stream.get("codec_name")} {stream.get("sample_rate")} Hz '
              f'{stream.get("channels")} ch')

    head(2, 'ASR CHUNKS AND THEIR REAL TIME BOUNDARIES')
    chunks = sorted((WORK / 'asr_chunks').glob('chunk_*.wav'))
    print(f'  chunk count : {len(chunks)}   (asr.py splits at 120 s)')
    offset = 0.0
    rows = []
    for index, path in enumerate(chunks):
        length = dur(path)
        rows.append((index, offset, offset + length, length, path.stat().st_size))
        offset += length
    print(f'  {"#":>3} {"start":>9} {"end":>9} {"length":>9} {"bytes":>10}')
    for index, start, end, length, size in rows:
        print(f'  {index:>3} {start:>9.3f} {end:>9.3f} {length:>9.3f} {size:>10}')
    print(f'  sum of chunk durations : {offset:.3f} s  (audio {audio_seconds:.3f} s, '
          f'delta {offset - audio_seconds:+.3f} s)')

    head(3, 'RECOGNISED SEGMENTS')
    text = asr_doc.get('text', '')
    print(f'  asr.json language      : {asr_doc.get("language")}')
    print(f'  asr.json text length   : {len(text)} characters')
    print(f'  asr.json segments[]    : {len(asr_doc.get("segments", []))}  '
          f'<- asr.py:69 always writes an empty list: ASR returns continuous text, not segments')
    alignment = load('alignment.json') or {}
    words = alignment.get('words') or []
    print(f'  alignment tokens       : {len(words)}  (first machine segmentation available)')
    print(f'  NOTE: the first *dialogue* segmentation appears only after merge_segments.')

    head(4, 'MAXIMUM end IN alignment.json')
    if words:
        ordered = sorted(words, key=lambda w: float(w['start']))
        max_end = max(float(w['end']) for w in words)
        min_start = min(float(w['start']) for w in words)
        print(f'  words        : {len(words)}')
        print(f'  min start    : {min_start:.3f} s')
        print(f'  max end      : {max_end:.3f} s')
        print(f'  audio        : {audio_seconds:.3f} s')
        print(f'  delta        : {max_end - audio_seconds:+.3f} s')
        print(f'  first token  : {ordered[0]["text"]!r} {ordered[0]["start"]}..{ordered[0]["end"]}')
        print(f'  last token   : {ordered[-1]["text"]!r} {ordered[-1]["start"]}..{ordered[-1]["end"]}')

    head(5, 'TIMESTAMPS BEYOND THE SOURCE AUDIO')
    offenders = []
    for word in words:
        if float(word['end']) > audio_seconds + AUDIO_TOL:
            offenders.append(('alignment', word.get('text'), word['start'], word['end']))
    lines = (load('segments.json') or {}).get('lines') or []
    for line in lines:
        if float(line['end']) > audio_seconds + AUDIO_TOL:
            offenders.append(('segments', line.get('line_id'), line['start'], line['end']))
    diar = load('diarization.json') or {}
    for turn in diar.get('turns') or []:
        if float(turn['end']) > audio_seconds + AUDIO_TOL:
            offenders.append(('diarization', turn.get('speaker_id'), turn['start'], turn['end']))
    print(f'  timestamps beyond {audio_seconds:.1f}s + {AUDIO_TOL}s tolerance: {len(offenders)}')
    for row in offenders[:10]:
        print(f'    {row}')
    if not offenders:
        print('  -> none: no artifact claims content outside the extracted audio')

    head(6, 'UNEXPLAINED GAPS AND OVERLAPS')
    print('  alignment (word level):')
    if words:
        ordered = sorted(words, key=lambda w: float(w['start']))
        gaps = []
        reach = float(ordered[0]['end'])
        for word in ordered[1:]:
            start = float(word['start'])
            if start - reach > GAP_ALIGN:
                gaps.append((round(reach, 3), round(start, 3), round(start - reach, 3)))
            reach = max(reach, float(word['end']))
        overlaps = sum(1 for a, b in zip(ordered, ordered[1:])
                       if float(b['start']) < float(a['end']) - 0.001)
        print(f'    gaps > {GAP_ALIGN:.0f}s : {len(gaps)}   total {sum(g[2] for g in gaps):.1f} s')
        for gap in gaps[:10]:
            print(f'      {gap}')
        print(f'    overlapping consecutive tokens : {overlaps}')
    print('  merged segments (line level):')
    if lines:
        line_gaps = [(round(float(a['end']), 3), round(float(b['start']), 3),
                      round(float(b['start']) - float(a['end']), 3))
                     for a, b in zip(lines, lines[1:]) if float(b['start']) - float(a['end']) > GAP_LINE]
        line_overlaps = [(a.get('line_id'), b.get('line_id'), round(float(a['end']) - float(b['start']), 3))
                         for a, b in zip(lines, lines[1:]) if float(b['start']) < float(a['end']) - 0.001]
        print(f'    gaps > {GAP_LINE:.0f}s : {len(line_gaps)}   total {sum(g[2] for g in line_gaps):.1f} s')
        for gap in line_gaps[:10]:
            print(f'      {gap}')
        print(f'    overlapping consecutive lines : {len(line_overlaps)}')
        for row in line_overlaps[:10]:
            print(f'      {row}')
    print('  diarization turns:')
    turns = sorted(diar.get('turns') or [], key=lambda t: float(t['start']))
    turn_overlaps = [(a.get('speaker_id'), b.get('speaker_id'),
                      round(float(a['end']) - float(b['start']), 3))
                     for a, b in zip(turns, turns[1:])
                     if float(b['start']) < float(a['end']) - 0.001]
    print(f'    turns total : {len(turns)}   overlapping consecutive turns : {len(turn_overlaps)}')
    for row in turn_overlaps[:10]:
        print(f'      {row}')

    head(7, 'SEGMENTS AFTER MERGING')
    if lines:
        durations = [float(l['end']) - float(l['start']) for l in lines]
        print(f'  merged lines   : {len(lines)}')
        print(f'  span           : {min(float(l["start"]) for l in lines):.3f} .. '
              f'{max(float(l["end"]) for l in lines):.3f} s')
        print(f'  line duration  : min {min(durations):.2f}  mean {sum(durations)/len(durations):.2f}  '
              f'max {max(durations):.2f} s')
        print(f'  zero-length    : {sum(1 for d in durations if d <= 0.001)}')
        print(f'  raw ASR chars covered : {sum(len(l.get("text","")) for l in lines)} '
              f'(asr.json text: {len(text)})')
        print(f'  first line     : {lines[0].get("line_id")} {lines[0]["start"]}..{lines[0]["end"]} '
              f'{lines[0].get("text","")[:30]!r}')
        print(f'  last line      : {lines[-1].get("line_id")} {lines[-1]["start"]}..{lines[-1]["end"]} '
              f'{lines[-1].get("text","")[:30]!r}')
        print(f'  schema keys    : {sorted(lines[0].keys())}')

    head(8, 'SPEAKERS FOUND / POSSIBLE MERGING OF DIFFERENT CHARACTERS')
    if diar:
        print(f'  model         : {diar.get("model")}')
        print(f'  bounds        : {diar.get("speaker_count_bounds")}')
        print(f'  speakers      : {diar.get("speaker_ids")}   count={diar.get("speaker_count")}')
        print(f'  turns         : {len(turns)}')
        span = {}
        for turn in turns:
            span[turn['speaker_id']] = span.get(turn['speaker_id'], 0.0) + (float(turn['end']) - float(turn['start']))
        for speaker, seconds in sorted(span.items(), key=lambda x: -x[1]):
            print(f'    {speaker}: {seconds:.1f} s of {audio_seconds:.1f} s '
                  f'({seconds / audio_seconds * 100:.1f}%)')
        if lines:
            counts = Counter(l.get('speaker') for l in lines)
            print(f'  merged lines per speaker : {dict(counts)}')
            top = max(counts.values()) / len(lines) * 100
            if top > 90:
                print(f'  SUSPICION: {top:.1f}% of lines carry one speaker label. For material with several')
                print('             speaking characters this indicates under-segmentation (different')
                print('             characters merged into one speaker), not a single-voice source.')
        print(f'  uncertain turns (<0.75s) : {len(diar.get("uncertain_turns") or [])}')
        print(f'  overlap intervals        : {len(diar.get("overlap_intervals") or [])}')
        for speaker, info in (diar.get('voice_types') or {}).items():
            print(f'    {speaker}: voice_type={info.get("voice_type")} confidence={info.get("confidence")} '
                  f'f0_median={info.get("f0_median_hz")} voiced={info.get("voiced_seconds")}s')

    head(9, 'ERRORS AND WARNINGS PER STAGE')
    for log in sorted(LOGS.glob('*.log')):
        text_log = log.read_text(errors='replace')
        lines_log = text_log.splitlines()
        bad = [l for l in lines_log if ('Traceback' in l or 'Error' in l or 'error' in l
                                        or 'WARN' in l or 'Warning' in l or 'WARNING' in l)]
        print(f'  {log.name}: {len(lines_log)} lines, {len(bad)} error/warning lines')
        for line in bad[:12]:
            print(f'      {line.strip()[:150]}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
