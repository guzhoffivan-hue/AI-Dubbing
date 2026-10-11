# Milestone: first real 20-minute dub (local)

Date: 2026-10-10. Result accepted by the project owner after listening.

## Deliverable

`test_run_1200s/output/test_dub_1200s.mp4` — 1200.000 s, 293 594 874 B,
video h264 1280x720 + audio aac 48 kHz stereo. Produced entirely on the local
Mac mini M4 (16 GB), no cloud API involved in any processing stage.

## Verified (machine-checked, not just "the script finished")

| check | value |
|---|---|
| MP4 duration / dub track | 1200.000 s / 1200.000 s |
| speech RMS at 10 / 50 / 90 % | -24.2 / -23.7 / -23.9 dBFS (even coverage) |
| last speech window | 1195 s of 1200 s -> 5 s silent tail |
| silent stretches > 40 s | none |
| overlapping lines | 0 (`timeline_validation.json`) |
| counters | segments 382 = translated 382 = TTS 382, 382 wav files |
| timeline bounds | first 0.00 s, last 1199.92 s, nothing beyond the audio |

## Stage chain actually used (each stage invoked directly; cli.py has no stage-subset flag)

```
extract_audio -> asr -> align -> diarize -> merge_segments      (16.0 + 1.2 + 10.4 min)
-> translate (Qwen3-8B-4bit via MLX/Metal)                      (46 min, 382 lines)
-> dialogue (speech budgets + validate_translation)
-> tts (Qwen3-TTS-0.6B MLX)                                     (22.8 min, 382 files)
-> fit_tts (atempo <= 1.7, then trim+fade)                      (6 s; 342 sped, 52 trimmed)
-> sfx timeline -> ambience -> mix (389 inputs, 56 s) -> mux    (14 s)
```

Run under `caffeinate`: without it the machine slept 76 of 108 minutes in an earlier run.

## Engine fixes applied during this work

| file | defect fixed |
|---|---|
| `tts.py` | restored from `tts.py.bak_vl`; the live file was a broken cache patch (SyntaxError) |
| `align.py` | chunk offsets from real chunk durations (was a magic 180 s vs 120 s chunks: +840 s of phantom timeline) |
| `common.py` | `chunk_offsets`, per-chunk text validation, proportional fallback, chunk-text resolution |
| `asr.py` | records per-chunk transcripts (`chunks` field) so alignment uses each chunk's own text |
| `dialogue.py` | refuses a partial/stale translation before TTS |
| `mix_audio.py` | validates the TTS timeline before mixing; restored working `amix` (broken `sidechaincompress` over 389 inputs made every real mix fail) |
| `fit_tts.py` | publishes the fitted duration so the manifest stays truthful |
| `translate.py` | budget from measured speech rate x tempo target (was a hard 13.5 chars/s forcing fact deletion); anti-deletion prompt; numbers rule; per-run glossary override |
| `merge_segments.py` | sentence-only breaks, real-pause threshold, zero-width tokens absorbed -> 1061 lines (mean 0.93 s, 54 zero-length) became 382 (mean 3.01 s, none) |
| `cli.py` | declared segment_merge artifact corrected |

Tests: 67 passing (`python3 -m unittest discover -s tests`).

Translation quality: v1 dropped facts in 14 % of lines and mistranslated numbers/names;
v3 (fixed policy + glossary override) dropped 9 %, and 十一月 -> "одиннадцать месяцев",
弹幕 -> "чат", 徐剑山 -> "Сюй Цзяньшань" are now correct.

## Known remaining issues

1. Mixer builds ONE ffmpeg filter graph with one input per line: proven at 389 inputs,
   but a 1-hour video is ~1100 and a 9-hour video ~10 000 -> needs block-wise mixing + concat.
2. 89 % of lines are compressed up to atempo 1.7 to fit their slot (52 also tail-trimmed);
   acceptable to the owner on this material, but it is a quality lever.
3. Diarization finds essentially one voice (SPEAKER_00 = 98.8 %): character voices are not
   separated. Diagnostic fragments `preflight/diagnostics/diag_98-110s.wav` and
   `diag_191-205s.wav` exist for a listening check.
4. ASR backend choice (Qwen3-ASR vs Whisper/MLX) is unvalidated: Whisper would be ~10-20x
   faster and yields word timestamps directly, but Chinese accuracy was not compared.
5. `sfx.py` / `ambience.py` read `ROOT/work/sfx_index.json` from a hard-coded path, which is
   why the isolated test root needs its own copy of the index.
6. `ai-dub-asr-check/` holds a 790 MB `ggml-large-v3-turbo.bin` unused by the pipeline.

## Full-video estimate (measured throughput)

~4.6 h of compute per 1 h of source video: ASR 50 min, diarization 31 min, translation 2.5 h,
TTS 1.3 h, rest ~10 min. A 9-hour film is therefore ~41 h (about two days) of continuous
compute, and only with idle sleep disabled. Whisper as ASR would remove ~8 h of that.

## Next steps (agreed priority)

1. ASR A/B on 2-3 existing 120 s chunks (cheap; decides the app's backend choice).
2. Block-wise mixing so long videos can be mixed at all.
3. 1-hour soak test to catch scaling problems before attempting the 9-hour source.
4. Either the user listens to the two diagnostic fragments (one narrator vs several voices),
   or accept single-voice dubbing for now.
5. After the engine is stable on a 1-hour run: adapters + state core in `AI-DUB-app`, then GUI.
