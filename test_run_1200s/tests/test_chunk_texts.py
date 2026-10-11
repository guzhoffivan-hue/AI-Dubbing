#!/usr/bin/env python3
"""Tests for per-chunk ASR text and its use by alignment.

Defect being pinned down: align.py used to cut the concatenated transcript into equal
character blocks (969 chars per chunk for the 1200 s test) while the real chunks held
975, 1008, 989, 975, 981, 948, 982, 949, 949, 935 characters. The text/audio boundaries
drifted by up to 82 characters (~10 s) and left untimed tails at chunk boundaries.

Run from the project root:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from common import (chunk_text_records, resolve_chunk_texts, split_text_proportionally,  # noqa: E402
                    validate_chunk_texts)

# The real per-chunk lengths measured in the 1200 s test (from the ASR log).
REAL_CHUNK_LENGTHS = [975, 1008, 989, 975, 981, 948, 982, 949, 949, 935]


def make_text(lengths):
    return ''.join(chr(0x4E00 + (index % 200)) * length for index, length in enumerate(lengths))


class BuildRecordTests(unittest.TestCase):
    def test_records_keep_order_text_and_length(self):
        texts = ['alpha', 'beta', 'gamma']
        records = chunk_text_records(texts, 'alphabetagamma')
        self.assertEqual([r['index'] for r in records], [0, 1, 2])
        self.assertEqual([r['text'] for r in records], texts)
        self.assertEqual([r['chars'] for r in records], [5, 4, 5])
        self.assertEqual(sorted(records[0].keys()), ['chars', 'index', 'text'])

    def test_empty_chunk_is_recorded_with_zero_length(self):
        records = chunk_text_records(['a', '', 'b'], 'ab')
        self.assertEqual([r['chars'] for r in records], [1, 0, 1])

    def test_mismatched_full_text_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            chunk_text_records(['a', 'b'], 'ac')
        self.assertIn('reconstruct the full transcript', str(caught.exception))

    def test_char_count_mismatch_is_rejected(self):
        records = [{'index': 0, 'text': 'abc', 'chars': 99}]
        with self.assertRaises(ValueError) as caught:
            validate_chunk_texts(records, 'abc', 1)
        self.assertIn('chars', str(caught.exception))

    def test_wrong_chunk_count_is_rejected(self):
        records = chunk_text_records(['a', 'b'], 'ab')
        with self.assertRaises(ValueError) as caught:
            validate_chunk_texts(records, 'ab', 3)
        self.assertIn('2 text chunks but 3 audio chunks', str(caught.exception))

    def test_out_of_order_records_are_rejected(self):
        records = [{'index': 1, 'text': 'a', 'chars': 1}, {'index': 0, 'text': 'b', 'chars': 1}]
        with self.assertRaises(ValueError) as caught:
            validate_chunk_texts(records, 'ab', 2)
        self.assertIn('out of order', str(caught.exception))

    def test_non_string_text_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_chunk_texts([{'index': 0, 'text': 5, 'chars': 1}], 'x', 1)


class ResolveTests(unittest.TestCase):
    def test_recorded_chunks_are_used_and_win_over_proportional_split(self):
        text = make_text(REAL_CHUNK_LENGTHS)
        records = chunk_text_records([text[sum(REAL_CHUNK_LENGTHS[:i]):sum(REAL_CHUNK_LENGTHS[:i + 1])]
                                      for i in range(len(REAL_CHUNK_LENGTHS))], text)
        texts, used = resolve_chunk_texts({'chunks': records}, text, len(REAL_CHUNK_LENGTHS))
        self.assertTrue(used)
        self.assertEqual(texts[3], text[sum(REAL_CHUNK_LENGTHS[:3]):sum(REAL_CHUNK_LENGTHS[:4])])
        self.assertEqual(''.join(texts), text)
        # the recorded split must differ from the equal-block split, which is the defect
        self.assertNotEqual(texts, split_text_proportionally(text, len(REAL_CHUNK_LENGTHS)))

    def test_missing_chunks_falls_back_without_claiming_accuracy(self):
        text = make_text(REAL_CHUNK_LENGTHS)
        texts, used = resolve_chunk_texts({}, text, len(REAL_CHUNK_LENGTHS))
        self.assertFalse(used)
        self.assertEqual(''.join(texts), text)
        self.assertEqual([len(t) for t in texts], [969, 969, 969, 969, 970, 969, 969, 969, 969, 969])

    def test_broken_chunks_field_raises_instead_of_falling_back(self):
        text = make_text(REAL_CHUNK_LENGTHS)
        with self.assertRaises(ValueError):
            resolve_chunk_texts({'chunks': [{'index': 0, 'text': 'x', 'chars': 1}]}, text,
                                len(REAL_CHUNK_LENGTHS))

    def test_proportional_split_loses_nothing(self):
        text = make_text(REAL_CHUNK_LENGTHS)
        texts = split_text_proportionally(text, len(REAL_CHUNK_LENGTHS))
        self.assertEqual(''.join(texts), text)
        self.assertEqual(len(texts), len(REAL_CHUNK_LENGTHS))

    def test_proportional_split_of_zero_parts(self):
        self.assertEqual(split_text_proportionally('abc', 0), [])


if __name__ == '__main__':
    unittest.main()
