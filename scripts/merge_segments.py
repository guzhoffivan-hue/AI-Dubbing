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

def absorb_zero_span_groups(groups):
    """Fold groups that carry text but no duration into a neighbouring same-speaker group.

    Some aligner tokens have start == end (761 of 8607 tokens in the 1200 s test, mostly
    function characters such as 的/了/一). A group consisting only of such tokens has a
    zero-length span; emitting it as a dialogue line creates a segment no TTS stage can
    fit. Folding it into an adjacent group of the SAME speaker preserves every token,
    every timestamp and the speaker assignment: nothing is invented, shifted or deleted.
    A token at a speaker boundary stays on its own, so confirmed speaker turns are never
    crossed.
    """
    groups = [list(group) for group in groups]
    folded = 0
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for index, group in enumerate(groups):
            if _span(group) > 0:
                continue
            speaker = _speaker(group)
            targets = [i for i in (index + 1, index - 1)
                       if 0 <= i < len(groups) and _speaker(groups[i]) == speaker]
            for target in targets:
                merged = (group + groups[target]) if target > index else (groups[target] + group)
                if _span(merged) > MAX_SEGMENT_DURATION:
                    # Refuse a fold that would itself break the duration bound: keeping a
                    # degenerate line is more honest than stretching a line over a gap.
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
