# AI-Dubbing — audit report (preserved evidence)

Snapshot taken **before** any Stage-1 edit: 2026-10-10.
Scope: read-only audit of `/Volumes/develop/AI-Dubbing` (old pipeline) and `/Volumes/develop/AI-DUB-app` (new core).
No file was modified, no dependency installed, no full pipeline executed, no `.env` contents read.

Companion files in this directory:

- `BASELINE_hashes.txt` — md5/size/mtime of every old-project script (live + backups), log and artifact at snapshot time.
- `media_metadata.txt` — ffprobe + partial md5 for all source/output media.
- `tts_version_proof.txt` — proof of which `tts.py` version produced the last working TTS run.
- `logs/` — verbatim copies of `AI-Dubbing/logs/*` (8 files).
- `work_json/` — verbatim copies of the run's JSON artifacts (27 files).

---

## 1. Headline

`output/test_dub.mp4` is a **1800-second container carrying a 43-segment Russian track that was produced for a 210-second window**. Speech cannot extend past 210 s because:

- `tts_manifest.json`: 43 lines, `max(tts_end) = 210.0025`;
- `mix_audio.py:11,42` takes `duration` from `audio_meta.json` (1800.0) and unconditionally stretches: `... apad,atrim=0:{duration}`;
- `mux.py:10` muxes with `-shortest`.

Direct measurement of the artifact (5 s RMS windows, mono 8 kHz decode):

| t, s | RMS, dBFS | interpretation |
|---|---|---|
| 0–205 | −22 … −25 | speech + ambience |
| 205–210 | −24 … −48 | last segment `SEG_0042` (207.68–210.00) |
| 210–1800 | −46 … −49 | ambience bed only, no speech |

Last window above −25 dBFS: **t = 205 s**. Reported symptom ("speech ends ≈03:30", i.e. 210 s) is reproduced from the artifact.

## 2. Root cause

**Stale cross-run artifact accepted as valid.** The fresh segment set was never translated, so the pipeline consumed the 43-line translation left over from the 210-second run.

Evidence chain:

1. `logs/42503442096-1-192_asr.log:1085` — `Created 1330 speaker-labelled lines; saved .../segments.json`
2. `logs/42503442096-1-192_asr.log:1089` — `Prepared 43 dialogue segments`, listing `SEG_0000 … SEG_0042` over `0.00–210.00`
3. `dialogue.py:6-18` emits exactly one output line per input line and reads **`translated_segments.json`** — so the 43 lines came from a stale file, not from the fresh 1330-line `segments.json`
4. Artifact key sets confirm two eras: `segments.json` uses `line_id='LINE_0001'…'LINE_1330'` (up to 2639.92 s); `dialogue_segments_raw.json`, `translation.json`, `tts_manifest.json` use `segment_id='SEG_0000'…'SEG_0042'` (up to 210.00 s)
5. Why no fresh translation existed — `logs/42503442096-1-192_translate.log:737-751`:
   `KeyError: 'source_text'` at `translate.py:37` (`build_glossary`), raised from `translate.py:53`.
   Cause: `merge_segments.py` was rewritten at 13:00 — it changed both the key schema (`line_id`/`speaker`/`text`) and the output filename (`dialogue_segments_raw.json` → `segments.json`), while `translate.py.bak_vl:47` still defaulted `--input` to `dialogue_segments_raw.json`
6. A later attempt on the fresh 1330 segments reached only line 600 and was killed; `translate.py:100-103` saves every 50 lines, so a **partial** file (600 = 12×50) is indistinguishable from a complete one
7. Nothing validated that the speech track covers the audio duration, so `mix_audio.py` silently padded 1590 s of ambience

## 3. Confirmed independent defects

| # | Location | Defect | Evidence |
|---|---|---|---|
| A5 | `align.py:32,41` | `CHUNK_SEC = 180.0` used as per-chunk offset while `asr.py:7` chunks at 120 s | 15 chunks of 120.000 s; coverage holes exactly at 180/360/…/2520; total 968.8 s of phantom gaps; `alignment.json` claims 2639.92 s on 1800 s of audio (+840 s) |
| A6 | `scripts/tts.py:94` | SyntaxError — a broken TTS disk-cache patch was pasted inside `write_json(WORK/'dialogue.json', …)`; shebang displaced to line 7 | AST parse fails at line 94; `tts.py.bak_vl` is identical except that patch (`tts_version_proof.txt`) |
| A7 | `cli.py:70` | `segment_merge` declares artifact `dialogue_segments_raw.json`, but `merge_segments.py:83` writes `segments.json` | resume/artifact check for that stage never matches |
| A4 | `translate.py:100-103` | intermediate saves overwrite the same file as the final save | 600-line partial accepted as complete |
| A8 | `asr.py:69` | `'segments': []` — ASR emits no timestamps | all timing depends on forced alignment |
| — | `cli.py:77-90` | `extract_audio`, `merge_segments`, `dialogue`, `sfx`, `ambience`, `mix_audio`, `mux` all log with `stage='asr'` | the "asr" log is a 1154-line history of nearly the whole pipeline |
| — | `align.py:37` | text is split across chunks proportionally by character count | no guaranteed text↔audio correspondence (hypothesis C1) |
| — | `align.py:54-63` | exception → silent linear per-character timing fallback | fabricated timings, no warning surfaced |

## 4. Artifact table (run `42503442096-1-192`, mode `test`, duration 1800.0)

| Stage | Artifact | Coverage | Count | State | Discrepancy |
|---|---|---|---|---|---|
| source | original video | 32 298.3 s (8 h 58 min) | — | **relocated** | found at `Download/42503442096-1-192.mp4`, size 7 026 950 160 **and** `mtime_ns` 1791326333813462138 match `status.json._signature` exactly |
| input/ | `input/42503442096-1-192.mp4` | 240.782 s | — | different file | 50 797 084 bytes — not the run's source |
| extract_audio | `audio.wav` | 1800.0 s | 15 chunks × 120.000 s | done | consistent |
| asr | `asr.json` | full 1800 s | text 14 438 chars | done | `segments: []` |
| alignment | `alignment.json` | **0 → 2639.92 s** | 12 796 words | done | +840 s, 968.8 s phantom gaps |
| diarization | `diarization.json` | 0.031 → 1799.969 s | 20 turns, 2 speakers | done | full coverage |
| segment_merge | `segments.json` | 0 → 2639.92 s | 1330 (`LINE_*`) | done | inherited inflated timebase |
| segment_merge (declared) | `dialogue_segments_raw.json` | 0 → 210.0 s | 43 (`SEG_*`) | **stale** | declared artifact ≠ written artifact (A7) |
| translation | `translated_segments.json` (run) | 0 → 210.0 s | 43 (`SEG_*`) | **stale** | should have matched 1330 `LINE_*` |
| translation | `translated_segments.json` (disk) | 0 → 982.08 s | 600 (`LINE_*`) | **partial 600/1330** | exactly 12×50 — killed at an intermediate save |
| dialogue | `dialogue.json` | 0 → 210.00 s | 43 | stale | files absent now, content in log |
| tts | `tts_manifest.json` | max `tts_end` 210.0025 | 43 | stale | 43 wavs (0000…0042), 43 resume entries |
| sfx | `sfx_timeline.json` | — | 0 events | done | — |
| ambience | `ambience.json` | looped to full duration | 1 candidate | done | fills 0–1800 s, masks the gap as "silence" |
| mix | `mix.wav` | 1800 s (`apad,atrim`) | 43 speech inputs | done | file absent now |
| mux | `output/test_dub.mp4` | **1800.000 s** | — | done | speech 0–205 s only |

Counter chain of the bad run:

```
audio 1800.0 s → ASR full (14 325 chars)
  → alignment 12 796 words to 2639.92 s      <- +840 s
  → segments.json 1330 lines to 2639.92 s
  → translated_segments.json 43 lines to 210.0 s   <- STALE
  → dialogue 43 → TTS 43 wav → mix (pad to 1800) → MP4 1800 s, speech 0-205 s
```

## 5. Version lineage (which implementation actually ran)

- `merge_segments.py` (live, 13:00) emits `SEG_`-free schema with `line_id`; `scripts.bak_1254` and `scripts.bak_arch` emit `segment_id='SEG_…'` and are the era that produced the 43 stale lines.
- `translate.py` (live, 18:36) added key normalisation and repointed `--input` to `segments.json`; `translate.py.bak_vl` (13:18) still read `dialogue_segments_raw.json`.
- `tts.py` (live) is unparseable; `tts.py.bak_vl` reproduces **43/43** `tts_resume.json` digests and differs from the live file only by the never-executed cache patch → it is the last working implementation.
- `asr.py` (live, 19:35) uses 120 s chunks; the run logged in `asr.log` used 180 s chunks, so the recorded 1800 s ASR predates the current `asr.py` while `asr_chunks/` (15×120 s) matches it — a second example of mixed-era artifacts.

## 6. Strategy decision (Part 5 conclusion)

Chosen: **strategy D** — `AI-DUB-app` as the controlling core; `AI-Dubbing` scripts as verified engines invoked through subprocess adapters in their own 3.11 virtualenvs, **after** a local repair and verification of the old pipeline.

Decisive argument: the old pipeline failed at the **orchestration layer** (cross-run artifact reuse, no consistency validation), which is exactly what the new core exists to prevent — but the new core as-is (`artifact_is_valid = exists() and size > 0`, `PipelineState.load` silently discarding state) would **not** have caught this defect, so hardening it is mandatory prerequisite work, not a bonus.

Hard constraint: old project runs on Python 3.11.13 (4 venvs: `asr`, `diarize`, `translate`, `tts`), new project on Python 3.14.8 — in-process unification is impossible; subprocess adapters are required. `common.py:7-9` already reads `AI_DUB_WORKDIR` / `AI_DUB_INPUT_VIDEO` / `AI_DUB_OUTPUT_DIR`, which is the ready-made adapter seam.

Explicit uncertainty: the old pipeline has **never** completed a full-length run (`logs/test_report.md`: "No full 8 h 58 min processing was run"), so throughput and memory behaviour on full length are unverified.
