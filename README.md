# AI-Dubbing

Offline, local-first video dubbing for Apple Silicon. `./dub` processes supported videos from `input/` sequentially, stores per-video intermediates in `work/<video>/`, and writes `<name>_ru.mp4` to `output/`. The final mux maps only the generated audio stream; the source audio is never mapped. The safety setting `use_original_audio` must remain `false`.

Heavy stages use separate Python environments so ASR, translation, and TTS model weights are not intentionally loaded in one process. A failed stage is recorded in `work/<video>/status.json`; rerunning `./dub` reuses completed artifacts for the same input and run mode. Detailed stage logs are written under `logs/`.

## Первый запуск

```sh
cd ~/AI-Dubbing
./setup
```

`./setup` creates isolated Python 3.11 environments, installs the local model dependencies and FFmpeg if needed, checks or builds the SFX index, then runs a bounded 150-second end-to-end sample. The pyannote diarization model is gated: accept its Hugging Face conditions and add a local token as `HF_TOKEN=...` in `.env` to run the sample. Setup never processes an entire long video. For a manual short diagnostic sample, use `./dub --test` (optionally `--start SEC --duration SEC`).

Python 3.11 and Homebrew are required for automatic environment setup. Model weights are downloaded into the local Hugging Face / MLX cache when their pipeline stage first runs; no inference is sent to a cloud service.

## Обычное использование

```sh
cp /path/to/video.mp4 ~/AI-Dubbing/input/
cd ~/AI-Dubbing
./dub
```

After processing:

```text
~/AI-Dubbing/output/
```

Several videos are processed one after another. A normal `./dub` run processes complete input videos; setup's sample is bounded. Do not use a full run until the sample result has been reviewed.

## Current implementation limits

The current SFX index is metadata and filename based; sound event/scene detection from video frames is not implemented yet. SFX placement uses conservative dialogue keyword matching. The configured TTS backend is Qwen3-TTS; setup does not yet benchmark Qwen3-TTS, Chatterbox, and CosyVoice or auto-select among them. The diarization stage requires a Hugging Face token and accepted model access. These items are reported as setup limitations rather than claimed as completed features.
