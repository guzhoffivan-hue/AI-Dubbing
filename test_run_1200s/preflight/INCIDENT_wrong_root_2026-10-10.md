# Incident report — wrongly-rooted staged run (2026-10-10 02:12–02:13)

Author of the incident: the agent. Recorded here in full before continuing, because it
touched preserved evidence in the live project.

## What happened

The staged-run watchdog `test_run_1200s/staged_run.py` resolved its project root with
`Path(__file__).resolve().parents[1]`. For a file at `AI-Dubbing/test_run_1200s/staged_run.py`,
`parents[1]` is `AI-Dubbing` — the **live project**, not the test root. As a result the
watchdog:

- started the **live** `scripts/cli.py` with `cwd` = the live project root;
- watched `AI-Dubbing/work/*/status.json` instead of the test root's work dir;
- therefore ran the pipeline in the live project, against `input/42503442096-1-192.mp4`
  (the 240.782 s clip), writing into the preserved run directory
  `work/42503442096-1-192/` of the 1800 s run.

Exposure window: **02:12:37 → ~02:13:1x** (killed on detection, process group terminated).
The run had reached `extract_audio: done`, `asr: running` (it had written the ASR chunks and
was loading the ASR model; the transcript was never written).

## Damage

| Path (live project) | Before | After | Recoverable from evidence? |
|---|---|---|---|
| `work/42503442096-1-192/audio.wav` | 57 600 078 B, 1800 s | 7 705 102 B, 240.782 s | No copy — **regenerated and verified** |
| `work/42503442096-1-192/audio_meta.json` | `duration: 1800.0` (md5 `e128379b`) | `duration: 240.782` | Evidence copy was lost to a basename collision — **reconstructed and verified** |
| `work/42503442096-1-192/status.json` | signature size 7 026 950 160 (md5 `f69fa9ea`) | reset signature (md5 `e2da8bb6`) | **Yes** — byte-exact restore |
| `work/42503442096-1-192/asr_chunks/chunk_0002.wav` | 3 840 044 B, 240–360 s | 25 068 B, 0.78 s | original bytes lost — **restored by proven-exact writer** |
| `logs/42503442096-1-192_asr.log` | md5 `e65caa7b` | appended (`aadc7f41`) | **Yes** — byte-exact restore |
| `logs/cli_stdout.log`, `logs/staged_run_wrapper.log` | did not exist | stray new files | left in place (no deletion) |

Verified **not** touched: `output/test_dub.mp4` (md5 `48eb4898`), `input/42503442096-1-192.mp4`
(md5 `2dc6c537`), `asr_chunks/chunk_0003…0014`, and every other JSON artifact in the run dir.

## Restoration, with verification

1. `status.json`, `logs/42503442096-1-192_asr.log` — restored from the byte-exact copies in
   `evidence/audit_2026-10-10/`; both md5s match the pre-incident baseline exactly.
2. `audio_meta.json` — reconstructed as
   `{"duration": 1800.0, "start": 0, "source": ".../input/42503442096-1-192.mp4"}` and verified
   against the baseline md5 `e128379b` → **byte-identical** (110 bytes).
3. `audio.wav` — regenerated with the project's own `scripts/extract_audio.py`
   (`--input` = untouched original, `--duration 1800 --start 0`). Result: size 57 600 078 B and
   `md5(first+last 4 MB)` = `0b55e951f2ccb4ec5be5c75196424de2`, matching the value recorded in
   `media_metadata.txt` at audit time **exactly**.
4. `asr_chunks/chunk_0000/0001` — proven undamaged: byte-identical to slices written by the
   exact code path `asr.py` uses (`sf.SoundFile.seek(idx*1920000); read(1920000); sf.write`).
5. `asr_chunks/chunk_0002` — restored with that same writer. The method was validated first on
   the **untouched** `chunk_0003`: the regenerated file reproduced its md5 `e6c74d31…`
   byte-for-byte. `chunk_0002` is an intermediate slice whose content is fully determined by
   `audio.wav` samples 240–360 s; its original bytes were not preserved anywhere, so this is a
   faithful reconstruction rather than a byte-restore.
6. Post-incident sweep: the live project now differs from `BASELINE_hashes.txt` **only** by the
   six intentional Stage-1 script repairs. Nothing else changed, nothing is missing.

## Root cause and the fix that prevents recurrence

- Root cause: `parents[1]` instead of resolving the file's own directory.
- Fix: `ROOT = Path(__file__).resolve().parent`, plus a `guard()` that refuses to start unless
  the resolved root is the isolated test root (not the live project), `.venvs` is the expected
  symlink, `input/` contains exactly `42503442096_first1200s.mp4`, and `work/` holds at most one
  run state. Added a `--dry-run` mode that prints resolved paths and the guard result.
- The corrected wrapper was dry-run and verified to resolve to
  `/Volumes/develop/AI-Dubbing/test_run_1200s` before the run was relaunched.

## Process lesson

Evidence preservation flattened two different files with the same basename
(`work/audio_meta.json` and `work/<slug>/audio_meta.json`) into one directory, silently losing
the run's copy. Future snapshots must mirror the source directory structure, not flatten it.
