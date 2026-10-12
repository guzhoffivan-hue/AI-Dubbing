# AI-Dubbing setup and bounded test

Status: **SETUP COMPLETE; short pipeline test completed.** No full-length processing was started.

## Test input and output

- Input: `input/42503442096-1-192.mp4`
- Source duration: 32,298.3 seconds (8 h 58 min)
- Tested only the first 150 seconds.
- Output: `output/test_dub.mp4` (150 seconds, 1280×720 H.264 video stream copied, AAC stereo at 48 kHz)
- WAV intermediate: `work/42503442096-1-192/mix.wav` (48 kHz stereo, peak 0.828, no clipped samples)
- ASR language: Chinese; one speaker detected; 19 translated/timed dialogue lines; 19 TTS files generated.
- Final mux maps `0:v:0` from the input and `1:a:0` from the generated mix. It does not map any source audio stream. `use_original_audio` is `false`.

## Local assets and performance

- SFX index: 347 local files; no library download was performed.
- TTS selected: `qwen3_mlx` (Qwen3-TTS 0.6B through MLX Audio). The local runtime completed synthesis. A three-backend comparison benchmark was not run.
- Last complete sample pipeline elapsed about 11.9 minutes, excluding model downloads performed during initial setup.
- Linear extrapolation to the full source is about 43 hours. This is a rough estimate; long-video throughput may differ.
- Peak per-process resident memory observed during setup was about 4.4 GiB. Total system/unified-memory peak was not captured.

## Problems and limits

- No SFX event met the current conservative keyword confidence rules, so the sample has no added SFX.
- The ambience selector now leaves ambience off when dialogue does not confidently identify a setting. This sample has no added ambience.
- Visual frame/scene analysis is not implemented; SFX and ambience selection are not yet scene-aware.
- Qwen3-TTS MLX Audio was used after the Torch/MPS Qwen3-TTS sampler produced NaNs. Chatterbox and CosyVoice3 were not benchmarked; their adapters still need validation.
- The Russian translation is technically generated, but review of the sample shows some inaccurate/awkward wording. The sample should be reviewed before a full episode.
- No full 8 h 58 min processing was run.
