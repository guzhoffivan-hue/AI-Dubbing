#!/usr/bin/env python3
"""Tests for the pre-mix integrity checks.

The headline test is ``test_incident_shape_tail_is_rejected``: it reproduces the shape
of the shipped defect (≈210 s of speech inside an 1800 s video) and asserts that the
pipeline now fails instead of building a video with no dubbing.

Run from the project root:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from integrity import IntegrityError, validate_translation, validate_tts_timeline  # noqa: E402


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return path


def write_wav(path, seconds, rate=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(seconds * rate))
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        block = b'\x00\x00' * rate
        while frames > 0:
            take = min(rate, frames)
            handle.writeframes(block[:take * 2])
            frames -= take
    return path


class BaseCase(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.work = Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()

    # --- builders ---------------------------------------------------------
    def build_segments(self, count=3, id_prefix='LINE', step=10.0):
        lines = [{'line_id': f'{id_prefix}_{index + 1:04d}', 'speaker': 'SPEAKER_00',
                  'start': round(index * step, 3), 'end': round(index * step + step - 1, 3),
                  'text': f'zh{index}'} for index in range(count)]
        write_json(self.work / 'segments.json', {'lines': lines})
        return lines

    def build_translation(self, segments, id_prefix=None, drop=0, text=None, shift=0.0):
        lines = []
        for index, segment in enumerate(segments[:len(segments) - drop]):
            segment_id = (f'{id_prefix}_{index + 1:04d}' if id_prefix
                          else (segment.get('segment_id') or segment.get('line_id')))
            translated = f'ru{index}' if text is None else text
            lines.append({'segment_id': segment_id, 'line_id': segment_id,
                          'speaker_id': 'SPEAKER_00',
                          'start': round(float(segment['start']) + shift, 3),
                          'end': round(float(segment['end']) + shift, 3),
                          'translate_text_source': 'test',
                          'translated_text': translated})
        write_json(self.work / 'translated_segments.json', {'lines': lines})
        return lines

    def build_tts(self, specs, audio_seconds, ids=None, count=None, manifest_ids=None,
                  manifest_shift=0.0, declared_override=None, break_paths=False,
                  empty_paths=False):
        """specs: list of (segment_id, start, duration). Returns (dialogue, manifest)."""
        dialogue, manifest = [], []
        for index, (segment_id, start, duration) in enumerate(specs):
            text = f'ru {segment_id}'
            dialogue.append({'segment_id': segment_id, 'speaker_id': 'SPEAKER_00',
                             'start': start, 'end': start + duration,
                             'available_end': start + duration,
                             'available_duration': duration, 'tts_speed': 1.0,
                             'overlap_confirmed': False, 'overlap_speaker_ids': [],
                             'translated_text': text})
            path = self.work / 'tts' / f'{segment_id}.wav'
            if empty_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'')
            else:
                write_wav(path, duration)
            if break_paths:
                path = self.work / 'tts' / 'absent' / f'{segment_id}.wav'
            declared = duration if declared_override is None else declared_override
            manifest_id = segment_id if manifest_ids is None else manifest_ids[index]
            manifest.append({'i': index, 'segment_id': manifest_id, 'speaker_id': 'SPEAKER_00',
                             'start': start + manifest_shift, 'end': start + duration,
                             'available_end': start + duration,
                             'available_duration': duration, 'tts_duration': declared,
                             'tts_end': start + declared, 'path': str(path),
                             'sample_rate': 8000, 'tts_speed': 1.0,
                             'overlap_confirmed': False, 'overlap_speaker_ids': [],
                             'translated_text': text, 'source_text': 'zh'})
        if count is not None:
            manifest = manifest[:count]
        write_json(self.work / 'dialogue.json', {'lines': dialogue})
        write_json(self.work / 'tts_manifest.json', {'backend': 'qwen3_mlx', 'lines': manifest})
        write_json(self.work / 'audio_meta.json',
                   {'duration': audio_seconds, 'start': 0.0, 'source': 'test.mp4'})
        return dialogue, manifest


class TranslationValidationTests(BaseCase):
    def test_complete_translation_is_accepted(self):
        segments = self.build_segments(count=3)
        self.build_translation(segments)
        segment_lines, translated_lines = validate_translation(self.work)
        self.assertEqual(len(segment_lines), 3)
        self.assertEqual(len(translated_lines), 3)

    def test_partial_translation_is_rejected(self):
        segments = self.build_segments(count=10)
        self.build_translation(segments, drop=5)
        with self.assertRaises(IntegrityError) as caught:
            validate_translation(self.work)
        self.assertIn('5 of 10', str(caught.exception))

    def test_translation_from_an_earlier_segment_set_is_rejected(self):
        segments = self.build_segments(count=3, id_prefix='LINE')
        self.build_translation(segments, id_prefix='SEG')
        with self.assertRaises(IntegrityError) as caught:
            validate_translation(self.work)
        self.assertIn('id mismatch', str(caught.exception))

    def test_translation_older_than_segments_is_rejected(self):
        segments = self.build_segments(count=3)
        self.build_translation(segments)
        # Rewrite the segment set after the translation, as segment_merge does on a rerun.
        self.build_segments(count=3)
        with self.assertRaises(IntegrityError) as caught:
            validate_translation(self.work)
        self.assertIn('earlier segment set', str(caught.exception))

    def test_empty_translated_text_is_rejected(self):
        segments = self.build_segments(count=2)
        self.build_translation(segments, text='')
        with self.assertRaises(IntegrityError) as caught:
            validate_translation(self.work)
        self.assertIn('translated_text is empty', str(caught.exception))

    def test_time_mismatch_is_rejected(self):
        segments = self.build_segments(count=2)
        self.build_translation(segments, shift=5.0)
        with self.assertRaises(IntegrityError) as caught:
            validate_translation(self.work)
        self.assertIn('time mismatch', str(caught.exception))

    def test_missing_translation_file_is_rejected(self):
        self.build_segments(count=2)
        with self.assertRaises(IntegrityError) as caught:
            validate_translation(self.work)
        self.assertIn('missing', str(caught.exception))

    def test_missing_segments_file_is_rejected(self):
        with self.assertRaises(IntegrityError) as caught:
            validate_translation(self.work)
        self.assertIn('segments.json', str(caught.exception))

    def test_segments_without_lines_is_rejected(self):
        write_json(self.work / 'segments.json', {'lines': []})
        write_json(self.work / 'translated_segments.json', {'lines': []})
        with self.assertRaises(IntegrityError) as caught:
            validate_translation(self.work)
        self.assertIn('no segments', str(caught.exception))


class TtsTimelineValidationTests(BaseCase):
    def test_complete_timeline_is_accepted(self):
        self.build_tts([('SEG_0000', 0.0, 5.0), ('SEG_0001', 5.0, 5.0)], audio_seconds=10.0)
        lines = validate_tts_timeline(self.work)
        self.assertEqual(len(lines), 2)

    def test_incident_shape_tail_is_rejected(self):
        # The shipped defect: speech for the first ~210 s inside an 1800 s container.
        self.build_tts([('SEG_0000', 0.0, 210.0)], audio_seconds=1800.0)
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        message = str(caught.exception)
        self.assertIn('no dubbing at all', message)
        self.assertIn('1590', message)

    def test_stale_manifest_is_rejected(self):
        # Same number of lines, same timings, but produced for a different segment set.
        self.build_tts([('LINE_0001', 0.0, 5.0), ('LINE_0002', 5.0, 5.0)],
                       audio_seconds=10.0, manifest_ids=['SEG_0000', 'SEG_0001'])
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        self.assertIn('stale manifest', str(caught.exception))

    def test_manifest_count_mismatch_is_rejected(self):
        self.build_tts([('SEG_0000', 0.0, 5.0), ('SEG_0001', 5.0, 5.0)],
                       audio_seconds=10.0, count=1)
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        self.assertIn('1 lines for 2 dialogue lines', str(caught.exception))

    def test_missing_audio_file_is_rejected(self):
        self.build_tts([('SEG_0000', 0.0, 5.0)], audio_seconds=5.0, break_paths=True)
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        self.assertIn('audio file is missing', str(caught.exception))

    def test_empty_audio_file_is_rejected(self):
        self.build_tts([('SEG_0000', 0.0, 5.0)], audio_seconds=5.0, empty_paths=True)
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        self.assertIn('audio file is empty', str(caught.exception))

    def test_declared_duration_mismatch_is_rejected(self):
        self.build_tts([('SEG_0000', 0.0, 5.0)], audio_seconds=200.0, declared_override=9.0)
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        self.assertIn('declared tts_duration', str(caught.exception))

    def test_line_beyond_audio_end_is_rejected(self):
        self.build_tts([('SEG_0000', 95.0, 10.0)], audio_seconds=100.0)
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        self.assertIn('beyond the', str(caught.exception))

    def test_internal_gap_is_rejected(self):
        self.build_tts([('SEG_0000', 0.0, 5.0), ('SEG_0001', 300.0, 5.0)], audio_seconds=310.0)
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        self.assertIn('hole', str(caught.exception))

    def test_leading_silence_is_rejected(self):
        self.build_tts([('SEG_0000', 200.0, 5.0)], audio_seconds=205.0)
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        self.assertIn('no dubbing', str(caught.exception))

    def test_non_monotonic_starts_are_rejected(self):
        self.build_tts([('SEG_0000', 100.0, 5.0), ('SEG_0001', 0.0, 5.0)], audio_seconds=105.0)
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        self.assertIn('before the previous line', str(caught.exception))

    def test_manifest_text_mismatch_is_rejected(self):
        dialogue, manifest = self.build_tts([('SEG_0000', 0.0, 5.0)], audio_seconds=5.0)
        manifest[0]['translated_text'] = 'a different line'
        write_json(self.work / 'tts_manifest.json', {'backend': 'qwen3_mlx', 'lines': manifest})
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        self.assertIn('does not match the dialogue text', str(caught.exception))

    def test_missing_manifest_is_rejected(self):
        write_json(self.work / 'dialogue.json', {'lines': [{'segment_id': 'SEG_0000'}]})
        write_json(self.work / 'audio_meta.json', {'duration': 5.0})
        with self.assertRaises(IntegrityError) as caught:
            validate_tts_timeline(self.work)
        self.assertIn('tts_manifest.json', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
