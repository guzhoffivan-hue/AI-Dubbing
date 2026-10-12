#!/usr/bin/env python3
"""Merge alignment, ASR text, and diarization into natural speech-budgeted segments (VideoLingo style)."""
import argparse
from common import WORK, read_json, write_json

# Only real sentence enders break a line. The Chinese comma '，' and the semicolon
# '；' are clause separators: breaking on them produced 684 one-clause lines out of
# 1061 in the 1200 s test (mean line 0.93 s), which starves translation of context
# and gives TTS slots smaller than any speakable Russian line.
SENTENCE_END = set('。！？!?…')
MAX_SEGMENT_DURATION = 6.5  # Upper bound, not a target: keeps a line safe for TTS.
# A real pause, not a micro-gap. In the 1200 s test only 14 of 1060 inter-line gaps
# reached 0.6 s, while 923 were below 0.25 s; the old 0.25 s threshold fragmented
# continuous speech into clause-sized pieces.
PAUSE_SPLIT = 0.6

def attach_punctuation(text, words):
    if not words:
        return words
    punct_after = [''] * len(words)
    cursor = 0
    for char in text:
        if cursor < len(words) and char == words[cursor].get('text'):
            cursor += 1
        else:
            idx = max(0, min(cursor - 1, len(words) - 1))
            punct_after[idx] += char
    for i, w in enumerate(words):
        w['punctuation_after'] = punct_after[i]
    return words

def assign_speakers(words, diar_segments):
    for w in words:
        mid = (w['start'] + w['end']) / 2.0
        spk = 'SPEAKER_00'
        for seg in diar_segments:
            if seg['start'] <= mid <= seg['end']:
                spk = seg.get('speaker_id') or seg.get('speaker') or spk
                break
        w['speaker'] = spk
    return words

def _bounds(group):
    """(start, end, has_timing) of the group's real speech.

    Zero-width aligner tokens carry text but no duration, so they must not stretch a
    line's span: a zero-width token sitting at a chunk boundary (600.0 s in the 1200 s
    test) would otherwise report a 14 s line for a few seconds of speech.
    """
    timed = [w for w in group if float(w['end']) > float(w['start'])]
    if not timed:
        # No duration anywhere in the group: report a zero-length position rather than
        # stretching a span across tokens that never occupy time.
        return float(group[0]['start']), float(group[0]['start']), False
    return float(timed[0]['start']), float(timed[-1]['end']), True

def _span(group):
    start, end, has_timing = _bounds(group)
    return end - start if has_timing else 0.0

def _speaker(group):
    return group[0].get('speaker', 'SPEAKER_00')

MIN_SEGMENT_SPAN = 0.15

def absorb_zero_span_groups(groups):
    """Fold zero-span and sub-150ms aligner micro-fragments into neighbouring same-speaker groups.

    Aligner tokens occasionally have start == end or micro-spans (< 150 ms, e.g. a single
    character like '行？' or '偷了！' aligned to 80 ms). Emitting them as isolated dialogue
    lines fragments sentences (e.g. '怎么不' split from '行？' leaving an unnatural silence
    at second 6) and starves TTS of speakable duration. Folding them into an adjacent group
    of the SAME speaker preserves all text and timestamps while restoring natural complete
    utterances.
    """
    groups = [list(group) for group in groups]
    folded = 0
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for index, group in enumerate(groups):
            span_val = _span(group)
            if span_val >= MIN_SEGMENT_SPAN:
                continue
            speaker = _speaker(group)
            if span_val <= 0.001:
                candidates = (index + 1, index - 1)
            else:
                prev_group = groups[index - 1] if index > 0 else None
                prev_punct = ''.join(x.get('punctuation_after', '') for x in prev_group) if prev_group else ''
                if prev_group and not any(c in prev_punct for c in SENTENCE_END):
                    candidates = (index - 1, index + 1)
                else:
                    candidates = (index + 1, index - 1)
            targets = [i for i in candidates
                       if 0 <= i < len(groups) and _speaker(groups[i]) == speaker]
            for target in targets:
                merged = (groups[target] + group) if target < index else (group + groups[target])
                if _span(merged) > MAX_SEGMENT_DURATION:
                    continue
                groups[target] = merged
                groups.pop(index)
                folded += 1
                changed = True
                break
            if changed:
                break
    return groups, folded


def group_words(words, voice_types=None):
    if not words:
        return []
    lines = []
    cur = []

    for w in words:
        if not cur:
            cur.append(w)
            continue
        prev = cur[-1]
        same_spk = (w.get('speaker') == prev.get('speaker'))
        gap = max(0.0, w['start'] - prev['end'])
        p_after = prev.get('punctuation_after', '')
        terminal = any(c in p_after for c in SENTENCE_END)
        cur_dur = w['end'] - cur[0]['start']

        # Break on: a confirmed speaker change, a real pause, a sentence end, or the
        # upper duration bound. A zero-width token never suppresses a break - the fold
        # below absorbs any group it ends up alone in, which keeps the duration bound
        # honest instead of letting a durationless token stretch a line to 14 s.
        should_break = (not same_spk) or gap >= PAUSE_SPLIT or terminal or cur_dur >= MAX_SEGMENT_DURATION
        if should_break:
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)

    if cur:
        lines.append(cur)

    lines, folded = absorb_zero_span_groups(lines)

    res = []
    for idx, chunk in enumerate(lines):
        text = ''.join(x.get('text', '') + x.get('punctuation_after', '') for x in chunk).strip()
        if not text:
            continue
        spk = chunk[0].get('speaker', 'SPEAKER_00')
        start, end, _ = _bounds(chunk)
        res.append({
            'line_id': f'LINE_{idx+1:04d}',
            'speaker': spk,
            'start': round(start, 3),
            'end': round(end, 3),
            'text': text
        })
    if folded:
        print(f'absorbed {folded} zero-duration groups into neighbouring same-speaker lines', flush=True)
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asr', default=str(WORK/'asr.json'))
    p.add_argument('--alignment', default=str(WORK/'alignment.json'))
    p.add_argument('--diarization', default=str(WORK/'diarization.json'))
    p.add_argument('--output', default=str(WORK/'segments.json'))
    a = p.parse_args()

    asr = read_json(a.asr)
    alignment = read_json(a.alignment)
    diarization = read_json(a.diarization)

    words = alignment.get('words', [])
    words = attach_punctuation(asr.get('text', ''), words)
    diar_list = diarization.get('turns') or diarization.get('segments') or []
    words = assign_speakers(words, diar_list)
    lines = group_words(words, diarization.get('voice_types', {}))

    write_json(a.output, {
        'language': asr.get('language', 'zh'),
        'speakers': sorted(list({x['speaker'] for x in lines})),
        'lines': lines
    })
    print(f'Created {len(lines)} speaker-labelled lines; saved {a.output}')

if __name__ == '__main__':
    main()
