#!/usr/bin/env python3
"""Tests for chunk time offsets and the align.py regression that caused them to drift.

Run from the project root:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from common import chunk_offsets  # noqa: E402

# The value asr.py used when the corrupted alignment was produced.
ASR_CHUNK_SECONDS = 120.0
# The magic constant align.py used as a per-chunk offset instead of the real duration.
LEGACY_MAGIC_CHUNK_SECONDS = 180.0
TOTAL_AUDIO_SECONDS = 1800.0


class ChunkOffsetTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(chunk_offsets([]), [])

    def test_single_chunk_starts_at_zero(self):
        self.assertEqual(chunk_offsets([120.0]), [0.0])

    def test_offsets_accumulate_real_durations(self):
        self.assertEqual(chunk_offsets([10.0, 5.5, 4.5]), [0.0, 10.0, 15.5])

    def test_fifteen_120s_chunks(self):
        offsets = chunk_offsets([ASR_CHUNK_SECONDS] * 15)
        self.assertEqual(len(offsets), 15)
        self.assertEqual(offsets[0], 0.0)
        self.assertEqual(offsets[1], 120.0)
        self.assertEqual(offsets[-1], 1680.0)

    def test_last_chunk_may_be_short(self):
        offsets = chunk_offsets([120.0, 120.0, 30.0])
        self.assertEqual(offsets, [0.0, 120.0, 240.0])

    def test_offsets_are_not_multiples_of_the_legacy_constant(self):
        offsets = chunk_offsets([ASR_CHUNK_SECONDS] * 15)
        self.assertNotEqual(offsets[1], LEGACY_MAGIC_CHUNK_SECONDS)
        legacy_offsets = [index * LEGACY_MAGIC_CHUNK_SECONDS for index in range(15)]
        self.assertNotEqual(offsets, legacy_offsets)

    def test_real_offsets_keep_the_timeline_inside_the_audio(self):
        offsets = chunk_offsets([ASR_CHUNK_SECONDS] * 15)
        real_end = offsets[-1] + ASR_CHUNK_SECONDS
        legacy_end = (15 - 1) * LEGACY_MAGIC_CHUNK_SECONDS + ASR_CHUNK_SECONDS
        self.assertLessEqual(real_end, TOTAL_AUDIO_SECONDS)
        # Regression: with the 180 s offset the timeline claimed 2640 s of content in a
        # 1800 s file, so everything past ~1220 s was pushed beyond the end of the video.
        self.assertGreater(legacy_end, TOTAL_AUDIO_SECONDS)
        self.assertAlmostEqual(legacy_end - TOTAL_AUDIO_SECONDS, 840.0)


class AlignSourceRegressionTests(unittest.TestCase):
    """Guard the fix itself: it is easy to reintroduce a magic chunk constant."""

    @classmethod
    def setUpClass(cls):
        cls.source_path = ROOT / 'scripts' / 'align.py'
        cls.source = cls.source_path.read_text()

    def test_align_parses(self):
        ast.parse(self.source)

    def test_align_uses_measured_chunk_durations(self):
        self.assertIn('chunk_offsets', self.source)
        self.assertIn('sf.info(str(p)).duration for p in chunk_files', self.source)

    def test_no_hard_coded_chunk_length_constant(self):
        """A numeric chunk-length constant is what silently corrupted the timeline."""
        tree = ast.parse(self.source)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Name) and 'CHUNK_SEC' in target.id
                            and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, (int, float))):
                        offenders.append(f'{target.id} = {node.value.value}')
        self.assertEqual(offenders, [], f'hard-coded chunk length reintroduced: {offenders}')

    def test_start_offset_is_not_a_product(self):
        """start_offset must come from measured durations, not from index * constant."""
        tree = ast.parse(self.source)
        products = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'start_offset' for target in node.targets):
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
                    products.append(ast.unparse(node.value))
        self.assertEqual(products, [], f'start_offset computed as a product: {products}')

    def test_start_offset_comes_from_the_offset_table(self):
        self.assertIn('offsets[idx]', self.source)

    def test_no_stale_proportional_split_variables(self):
        """The proportional split left 'cursor' behind in the loop and crashed alignment."""
        names = {node.id for node in ast.walk(ast.parse(self.source)) if isinstance(node, ast.Name)}
        self.assertNotIn('cursor', names, 'stale cursor variable from the proportional split remains')
        self.assertNotIn('avg_chunk_chars', names, 'stale proportional split remains')
        self.assertIn('chunk_texts', names)

    def test_each_chunk_uses_its_own_text(self):
        self.assertIn('part_text = chunk_texts[idx]', self.source)
        self.assertIn('resolve_chunk_texts', self.source)


if __name__ == '__main__':
    unittest.main()
