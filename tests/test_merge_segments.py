#!/usr/bin/env python3
"""Tests for dialogue grouping in merge_segments.py.

The 1200 s test produced 1061 lines (mean 0.93 s, 54 of them zero-length) because a
line was broken on every Chinese comma and on every 0.25 s micro-gap. These tests pin
the corrected rules: break on sentence end, real pause, confirmed speaker change or the
duration bound - and never emit a zero-length line when a same-speaker neighbour exists.

Run from the project root:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from merge_segments import PAUSE_SPLIT, MAX_SEGMENT_DURATION, assign_speakers, group_words, absorb_zero_span_groups  # noqa: E402


def tok(text, start, end, speaker='SPEAKER_00', punct=''):
    return {'text': text, 'start': float(start), 'end': float(end),
            'speaker': speaker, 'punctuation_after': punct}


def span(line):
    return float(line['end']) - float(line['start'])


class BreakRuleTests(unittest.TestCase):
    def test_comma_does_not_break(self):
        words = [tok('甲', 0, 1, punct='，'), tok('乙', 1, 2, punct='，'), tok('丙', 2, 3)]
        lines = group_words(words)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]['text'], '甲，乙，丙')

    def test_micro_gap_does_not_break_but_real_pause_does(self):
        closed = group_words([tok('甲', 0, 1), tok('乙', 1.25, 2)])
        self.assertEqual(len(closed), 1, 'a 0.25 s micro-gap must not split a line')
        split = group_words([tok('甲', 0, 1), tok('乙', 1 + PAUSE_SPLIT, 3)])
        self.assertEqual(len(split), 2, 'a real pause must split a line')

    def test_sentence_end_breaks(self):
        words = [tok('甲', 0, 1, punct='。'), tok('乙', 1, 2), tok('丙', 2, 3, punct='？'), tok('丁', 3, 4)]
        lines = group_words(words)
        self.assertEqual([l['text'] for l in lines], ['甲。', '乙丙？', '丁'])

    def test_speaker_change_breaks(self):
        words = [tok('甲', 0, 1, speaker='SPEAKER_00'), tok('乙', 1, 2, speaker='SPEAKER_01')]
        lines = group_words(words)
        self.assertEqual(len(lines), 2)
        self.assertEqual([l['speaker'] for l in lines], ['SPEAKER_00', 'SPEAKER_01'])

    def test_duration_bound_is_respected(self):
        words = [tok(str(i), i, i + 1) for i in range(12)]
        lines = group_words(words)
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(span(line), MAX_SEGMENT_DURATION + 1.0)


class ZeroWidthTests(unittest.TestCase):
    def test_zero_width_token_does_not_open_a_line(self):
        # '。' breaks, and the next token has no duration: it must not become a
        # zero-length line, and it must not stretch the following line's span.
        words = [tok('甲', 0, 1, punct='。'), tok('的', 1, 1), tok('乙', 1, 2)]
        lines = group_words(words)
        self.assertEqual([l['text'] for l in lines], ['甲。', '的乙'])
        self.assertEqual(sum(1 for l in lines if span(l) <= 0.001), 0)

    def test_zero_width_token_at_a_distant_boundary_does_not_stretch_a_line(self):
        # A zero-width token parked at a chunk boundary (120 s) must not turn a 6 s
        # line of speech into a 120 s line.
        words = [tok('甲', 0, 1), tok('乙', 1, 6), tok('丙', 120.0, 120.0)]
        lines = group_words(words)
        self.assertEqual(''.join(l['text'] for l in lines), '甲乙丙')
        for line in lines:
            self.assertLessEqual(span(line), MAX_SEGMENT_DURATION)

    def test_zero_width_token_at_speaker_boundary_is_not_crossed(self):
        words = [tok('甲', 0, 1, speaker='SPEAKER_00'),
                 tok('的', 1, 1, speaker='SPEAKER_01'),
                 tok('乙', 1, 2, speaker='SPEAKER_00')]
        lines = group_words(words)
        self.assertEqual([l['speaker'] for l in lines], ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_00'])

    def test_fold_keeps_text_and_speaker(self):
        groups = [[tok('甲', 0, 1)], [tok('的', 1, 1)], [tok('乙', 2, 3)]]
        folded, count = absorb_zero_span_groups(groups)
        self.assertEqual(count, 1)
        # The zero-width group is folded into the FOLLOWING same-speaker group, so the
        # text order and the timestamps are unchanged.
        self.assertEqual([len(g) for g in folded], [1, 2])
        self.assertEqual(''.join(w['text'] for g in folded for w in g), '甲的乙')
        self.assertEqual(folded[1][0]['start'], 1.0)


class InvariantTests(unittest.TestCase):
    def setUp(self):
        self.words = [tok('甲', 0, 1, punct='，'), tok('的', 1, 1), tok('乙', 1, 2, punct='。'),
                      tok('丙', 2, 2), tok('丁', 2, 3, speaker='SPEAKER_01', punct='！'),
                      tok('戊', 3, 4, speaker='SPEAKER_01')]

    def test_text_is_never_lost(self):
        lines = group_words(self.words)
        expected = ''.join(w['text'] + w['punctuation_after'] for w in self.words)
        self.assertEqual(''.join(l['text'] for l in lines), expected)

    def test_timestamps_are_taken_from_tokens_only(self):
        token_starts = {w['start'] for w in self.words}
        token_ends = {w['end'] for w in self.words}
        for line in group_words(self.words):
            self.assertIn(float(line['start']), token_starts)
            self.assertIn(float(line['end']), token_ends)

    def test_speaker_ids_are_preserved(self):
        lines = group_words(self.words)
        self.assertEqual({l['speaker'] for l in lines}, {'SPEAKER_00', 'SPEAKER_01'})

    def test_no_line_spans_two_speakers(self):
        # Disjoint token intervals make containment unambiguous; zero-width tokens are
        # covered by test_zero_width_token_at_speaker_boundary_is_not_crossed instead.
        words = [tok('甲', 0, 1), tok('乙', 1, 2, punct='。'),
                 tok('丙', 2, 3, speaker='SPEAKER_01'),
                 tok('丁', 3, 4, speaker='SPEAKER_01')]
        lines = group_words(words)
        self.assertEqual([l['speaker'] for l in lines], ['SPEAKER_00', 'SPEAKER_01'])
        for line in lines:
            members = [w['speaker'] for w in words
                       if float(w['start']) >= float(line['start'])
                       and float(w['end']) <= float(line['end'])]
            self.assertEqual(len(set(members)), 1)
            self.assertGreater(len(members), 0)

    def test_grouping_is_deterministic(self):
        self.assertEqual(group_words(self.words), group_words(self.words))


class AssignSpeakerTests(unittest.TestCase):
    def test_collar_word_snaps_to_nearest_turn_instead_of_speaker_00(self):
        words = [
            {'text': '你', 'start': 9.90, 'end': 9.98, 'punctuation_after': ''},
            {'text': '好', 'start': 10.00, 'end': 10.40, 'punctuation_after': '。'},
        ]
        turns = [{'speaker_id': 'SPEAKER_01', 'start': 10.00, 'end': 12.00}]
        assigned = assign_speakers(words, turns)
        self.assertEqual([w['speaker'] for w in assigned], ['SPEAKER_01', 'SPEAKER_01'])

    def test_intra_clause_micro_blip_is_smoothed(self):
        words = [
            {'text': '我', 'start': 1.0, 'end': 1.3, 'punctuation_after': ''},
            {'text': '们', 'start': 1.3, 'end': 1.5, 'punctuation_after': ''},
            {'text': '去', 'start': 1.5, 'end': 1.9, 'punctuation_after': '。'},
        ]
        turns = [
            {'speaker_id': 'SPEAKER_00', 'start': 0.8, 'end': 1.31},
            {'speaker_id': 'SPEAKER_01', 'start': 1.31, 'end': 1.49},
            {'speaker_id': 'SPEAKER_00', 'start': 1.49, 'end': 2.5},
        ]
        assigned = assign_speakers(words, turns)
        self.assertEqual([w['speaker'] for w in assigned], ['SPEAKER_00', 'SPEAKER_00', 'SPEAKER_00'])


if __name__ == '__main__':
    unittest.main()
