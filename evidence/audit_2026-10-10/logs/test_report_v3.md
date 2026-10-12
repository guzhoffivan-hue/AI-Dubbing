# Test dub v3 report

- Output: `output/test_dub_v3.mp4`
- Source: `input/42503442096-1-192.mp4`, first 150 seconds only
- Processing time: 173.6 seconds (2 min 54 sec)
- Video: 150 seconds, H.264; audio: AAC, stereo, 48 kHz
- ASR, alignment, and translation were reused and not rerun.

## Speaker and voice assignment

| Speaker ID | Segments | Acoustic voice type | Confidence | Voice ID |
| --- | ---: | --- | --- | --- |
| `SPEAKER_00` | 19 | female | medium | `VOICE_00` |

Pyannote returned one continuous turn from about 0.03 to 149.97 seconds and one speaker ID. It did not identify separate character voices in this sample. The acoustic voice estimate used pYIN pitch over the diarized audio: median 205.3 Hz, Q25 191.6 Hz, Q75 223.9 Hz, and 2.26 seconds of confidently voiced frames. Ambiguous or shorter estimates are classified as `unknown`.

All 19 dialogue segments preserve `speaker_id` and `voice_id`. The TTS manifest contains 19 synthesized segments, each mapped to `SPEAKER_00 → VOICE_00`; no segment changed profiles and no fallback was used.

## Voice-profile method

The configured backend remains `qwen3_mlx`. One native Russian female voice profile was designed once and saved at `work/42503442096-1-192/voice_profiles/voice_00.wav`. The same reference audio and text were used to synthesize every segment for that speaker, with a stable speaker-specific sampling seed. The source Chinese audio was used for acoustic voice-type estimation only, not as a cloning reference, to avoid reintroducing the Chinese accent. The profile map is saved in `work/42503442096-1-192/voice_profiles.json`.

## Remaining limitation

This test contains only one diarized speaker. If the source section is expected to contain separate audible characters, speaker separation remains a diarization limitation: this run cannot validate assignments for additional speakers or genders. The female estimate has medium confidence because only 2.26 seconds of voiced pitch frames passed the conservative voicing threshold. Voice identity consistency is enforced by using one saved reference per speaker; subjective listening is still needed to judge perceived similarity across the finished lines.
