#!/usr/bin/env python3
"""Guard against a live stage script being left in an unrunnable state.

scripts/tts.py was once broken by a partially applied patch: a block was pasted inside
a write_json(...) call, which made the file fail to parse. The whole TTS stage was dead
and nothing detected it. This test keeps that from happening unnoticed again.

Run from the project root:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'

# Every stage script the CLI invokes, in pipeline order.
LIVE_SCRIPTS = ['extract_audio.py', 'asr.py', 'align.py', 'diarize.py', 'merge_segments.py',
                'translate.py', 'dialogue.py', 'tts.py', 'sfx.py', 'ambience.py',
                'mix_audio.py', 'mux.py']

# Modules that the stage scripts import at runtime and that must stay importable.
SUPPORT_SCRIPTS = ['common.py', 'integrity.py']

# The TTS version that produced the last working run, identified by the resume-digest
# tag it writes into tts_resume.json.
TTS_RESUME_TAG = '_fixed_profile_v2'


class LiveScriptsParseTests(unittest.TestCase):
    def test_expected_scripts_exist(self):
        for name in LIVE_SCRIPTS + SUPPORT_SCRIPTS:
            with self.subTest(script=name):
                self.assertTrue((SCRIPTS / name).exists(), f'{name} is missing')

    def test_every_live_script_parses(self):
        for name in LIVE_SCRIPTS + SUPPORT_SCRIPTS:
            with self.subTest(script=name):
                ast.parse((SCRIPTS / name).read_text(), filename=name)

    def test_tts_still_has_the_proven_resume_semantics(self):
        source = (SCRIPTS / 'tts.py').read_text()
        self.assertIn(TTS_RESUME_TAG, source, 'tts.py is not the version proven against tts_resume.json')

    def test_tts_has_a_shebang_on_the_first_line(self):
        first_line = (SCRIPTS / 'tts.py').read_text().splitlines()[0]
        self.assertTrue(first_line.startswith('#!'), f'first line is {first_line!r}')

    def test_integrity_is_wired_into_the_mixer_and_dialogue_builder(self):
        self.assertIn('validate_tts_timeline', (SCRIPTS / 'mix_audio.py').read_text())
        self.assertIn('validate_translation', (SCRIPTS / 'dialogue.py').read_text())


if __name__ == '__main__':
    unittest.main()
