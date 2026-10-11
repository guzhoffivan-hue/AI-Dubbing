# Evidence snapshot — AI-Dubbing audit, 2026-10-10

Taken **before** any Stage-1 repair edit. Nothing here was moved from its original
location; everything is a copy. Originals remain in place.

| File / dir | What it is |
|---|---|
| `AUDIT_REPORT.md` | Full audit report: pipeline chain, root cause of the 03:30 speech cutoff, artifact table, defect classification, strategy decision (D). |
| `BASELINE_hashes.txt` | md5 + size + mtime of every old-project script (live + 13 backups + 2 backup trees), log, and JSON artifact, as of the snapshot. Use this to verify exactly what Stage 1 changed. |
| `media_metadata.txt` | ffprobe + partial md5 (first+last 4 MB) for source video, current `input/` file, `audio.wav`, all 6 `output/*.mp4`, chunk inventory, TTS wav inventory. Full md5 was deliberately skipped for the 7 GB original. |
| `tts_version_proof.txt` | Proof that `scripts/tts.py.bak_vl` is the last working TTS implementation, plus the full diff against the broken live `scripts/tts.py`, plus a copy of the broken file. |
| `logs/` | Verbatim copies of `logs/*.log` and `logs/*.md` (8 files) — the primary forensic record. |
| `work_json/` | Verbatim copies of the run's JSON artifacts (27 files): `status.json`, `segments.json`, `translated_segments.json`, `tts_manifest.json`, `tts_resume.json`, `voice_profiles.json`, `alignment.json`, `diarization.json`, `asr.json`, `ambience.json`, `timeline_validation.json`, etc. |

## Key preserved facts

- Original source video located: `/Volumes/develop/Download/42503442096-1-192.mp4`,
  7 026 950 160 bytes, `mtime_ns` 1791326333813462138, duration 32 298.300 s.
  Both size and `mtime_ns` match `work/42503442096-1-192/status.json` → `_signature` exactly.
- `input/42503442096-1-192.mp4` is a **different** file (50 797 084 bytes, 240.782 s).
- The delivered artifact `output/test_dub.mp4` (1800.000 s) contains speech only in 0–205 s;
  from 210 s to the end it is the ambience bed at −46…−49 dBFS.
- `scripts/tts.py` as of the snapshot cannot be parsed (SyntaxError, line 94).

## Secrets

No `.env`, token or key file was read, copied, or printed at any point.
