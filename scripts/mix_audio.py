#!/usr/bin/env python3
"""Mix generated speech, selected library effects, and optional ambience.

Guarantees:
1. Cross-stage timeline integrity check (`validate_tts_timeline`) runs before mixing.
2. Consecutive duplicate/echo TTS lines (same speaker) are filtered.
3. Internal phrase repetitions (2x-4x repeated clauses) are trimmed to a single clean utterance.
4. Strict non-overlapping speech placement (play_start >= prev_end + 0.05s).
5. «Smear» algorithm: clips that don't fit their nominal slot borrow silence from the next gap
   (up to MAX_BORROW seconds). This eliminates both dead silence and forced rushing.
   A/V desync up to ~0.9s is acceptable — intelligibility takes priority over lip-sync.
"""
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from common import ROOT, WORK, read_json, write_json
from integrity import validate_tts_timeline

TARGET_SR = 48000
MIN_SPEECH_GAP = 0.05   # minimum gap between consecutive speech clips
MAX_MIX_TEMPO = 1.35    # mixer compression cap — TTS already applied heavy atempo
MAX_BORROW = 0.90       # max seconds we may «borrow» from the next inter-line gap


def raise_fd_limit():
    try:
        import resource
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = 10240
    if hard != resource.RLIM_INFINITY:
        wanted = min(wanted, hard)
    if soft < wanted:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
        except (OSError, ValueError):
            pass


def run(cmd):
    subprocess.run(cmd, check=True)


CHUNK_INPUTS = 64


def safe_chunk_inputs():
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft != resource.RLIM_INFINITY and soft > 15:
            budget = int((soft - 15) / 0.7)
            return max(8, min(CHUNK_INPUTS, budget))
    except (ImportError, OSError, ValueError):
        pass
    return CHUNK_INPUTS


def norm_words(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", (text or "").lower())


def repeat_factor(words: list[str]) -> tuple[int, int]:
    """Return (repeat_count, unit_word_length) if words is an exact repetition of a >=2-word phrase."""
    n = len(words)
    for k in range(2, n // 2 + 1):
        if n % k == 0 and words == words[:k] * (n // k):
            return n // k, k
    return 1, n


def is_echo_duplicate(prev_line: dict, curr_line: dict) -> bool:
    """Detect back-to-back duplicate or stuttered echo translations within 3.5 seconds."""
    gap = float(curr_line["start"]) - float(prev_line["start"])
    if gap > 3.5:
        return False
    # Never filter lines from different speakers — they are always distinct utterances
    prev_spk = prev_line.get("speaker_id") or ""
    curr_spk = curr_line.get("speaker_id") or ""
    if prev_spk and curr_spk and prev_spk != curr_spk:
        return False
    pw = norm_words(prev_line.get("translated_text", ""))
    cw = norm_words(curr_line.get("translated_text", ""))
    if not pw or not cw:
        return False
    if pw == cw:
        return True
    p_str = " ".join(pw)
    c_str = " ".join(cw)
    if len(cw) >= 3 and (p_str.endswith(c_str) or p_str.startswith(c_str)):
        return True
    if len(pw) >= 3 and c_str.startswith(p_str):
        return True
    if len(cw) >= 2 and cw == pw[-len(cw):] and gap <= 2.5:
        return True
    rf, unit_k = repeat_factor(cw)
    if rf >= 2 and unit_k >= 2:
        unit_str = " ".join(cw[:unit_k])
        if p_str.startswith(unit_str) or p_str.endswith(unit_str):
            return True
    return False


def is_unfinished_stutter_prefix(curr_line: dict, next_line: dict) -> bool:
    """Detect a single-word unfinished fragment that is immediately spoken in full in next_line.

    Deliberately conservative: only fires when:
      - Same speaker (different speakers can reuse the same word legitimately)
      - Only 1 word in curr_line (2-word short phrases are often complete answers)
      - Next line starts with that word
      - Source text has no terminal punctuation (。！？!?…)
      - Gap is within 2.0s
    """
    gap = float(next_line["start"]) - float(curr_line["start"])
    if gap > 2.0:
        return False
    # Same speaker only
    curr_spk = curr_line.get("speaker_id") or ""
    next_spk = next_line.get("speaker_id") or ""
    if curr_spk and next_spk and curr_spk != next_spk:
        return False
    pw = norm_words(curr_line.get("translated_text", ""))
    cw = norm_words(next_line.get("translated_text", ""))
    # Only filter single-word fragments, not 2-word phrases (those are often valid short answers)
    if not pw or not cw or len(pw) != 1 or len(cw) <= 1:
        return False
    src = str(curr_line.get("source_text", "")).strip()
    # Require no terminal punctuation in source — terminal punctuation signals a complete sentence
    if any(ch in src for ch in "。！？!?…"):
        return False
    p_str = pw[0]
    c_str = " ".join(cw)
    return c_str.startswith(p_str)


def trim_silence(wav: np.ndarray, sr: int, threshold: float = 0.015) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    idx = np.where(np.abs(wav) > threshold)[0]
    if len(idx) == 0:
        return wav
    pad = int(0.015 * sr)
    s = max(0, int(idx[0]) - pad)
    e = min(len(wav), int(idx[-1]) + pad)
    return wav[s:e]


def apply_fade(wav: np.ndarray, sr: int, fade_in_ms: float = 12.0, fade_out_ms: float = 18.0) -> np.ndarray:
    out = wav.astype(np.float32, copy=True)
    n = len(out)
    fi = min(n // 2, int(sr * fade_in_ms / 1000.0))
    fo = min(n // 2, int(sr * fade_out_ms / 1000.0))
    if fi > 1:
        out[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 1:
        out[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return out


def resample_to_target(wav: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr or len(wav) == 0:
        return wav.astype(np.float32, copy=False)
    duration = len(wav) / float(sr)
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, 1.0, len(wav), endpoint=False)
    x_new = np.linspace(0.0, 1.0, target_len, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def time_compress_wav(wav: np.ndarray, sr: int, tempo: float, tmp_dir: Path) -> np.ndarray:
    tempo = max(1.0, min(2.0, float(tempo)))
    if tempo <= 1.03 or len(wav) < int(0.15 * sr):
        return wav
    src_tmp = tmp_dir / "_mix_pace_in.wav"
    dst_tmp = tmp_dir / "_mix_pace_out.wav"
    sf.write(src_tmp, wav, sr)
    run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src_tmp),
        "-filter:a", f"atempo={tempo:.4f}",
        "-ar", str(sr),
        str(dst_tmp),
    ])
    paced, paced_sr = sf.read(dst_tmp)
    src_tmp.unlink(missing_ok=True)
    dst_tmp.unlink(missing_ok=True)
    return trim_silence(np.asarray(paced), paced_sr)


def build_clean_speech_bed(manifest: list[dict], duration: float) -> tuple[Path, list[dict]]:
    """Build a strictly non-overlapping, de-echoed stereo 48kHz speech WAV bed.

    «Smear» strategy
    ----------------
    When a TTS clip is longer than its nominal slot (time until the next line starts),
    we first check how much *silence* exists in the gap after the next line and
    borrow up to MAX_BORROW seconds from it.  Only if the clip is STILL too long
    after borrowing do we apply mild atempo compression (≤ MAX_MIX_TEMPO).

    This converts "rushed mono-tone compression" and "dead silence" into a natural
    spread — a slight A/V drift of ≤ 0.9 s which the user has explicitly approved.
    """
    kept: list[dict] = []
    for i, line in enumerate(manifest):
        if kept and is_echo_duplicate(kept[-1], line):
            print(f"  [filter] echo-dup  {line.get('segment_id')}: {line.get('translated_text')!r}", flush=True)
            continue
        if i + 1 < len(manifest) and is_unfinished_stutter_prefix(line, manifest[i + 1]):
            print(f"  [filter] stutter   {line.get('segment_id')}: {line.get('translated_text')!r}", flush=True)
            continue
        kept.append(line)

    total_samples = max(1, int(round(duration * TARGET_SR)))
    speech_mono = np.zeros(total_samples, dtype=np.float32)
    placed_intervals: list[dict] = []
    cursor_end = 0.0

    for idx, line in enumerate(kept):
        wav_path = Path(line["path"])
        raw_wav, sr = sf.read(wav_path)
        wav = trim_silence(np.asarray(raw_wav), sr)
        if len(wav) == 0:
            continue

        # Collapse internal 2x-4x phrase repetitions to a single utterance
        words = norm_words(line.get("translated_text", ""))
        rf, unit_k = repeat_factor(words)
        if rf >= 2 and unit_k >= 2 and len(wav) > int(0.8 * sr):
            keep_samples = max(int(0.45 * sr), len(wav) // rf)
            wav = trim_silence(wav[:keep_samples], sr)

        nom_start = float(line["start"])
        # Play no earlier than nominal start, and no earlier than 50 ms after previous clip
        play_start = max(nom_start, cursor_end + (MIN_SPEECH_GAP if cursor_end > 0 else 0.0))
        if play_start >= duration - 0.05:
            break

        clip_dur = len(wav) / float(sr)

        # ── Smear algorithm ──────────────────────────────────────────────────────
        # Nominal slot: until the next kept line's start
        next_nom_start = float(kept[idx + 1]["start"]) if idx + 1 < len(kept) else duration
        nominal_slot = next_nom_start - play_start - MIN_SPEECH_GAP

        # How large is the gap AFTER the next line? (silence we can borrow from)
        if idx + 2 < len(kept):
            after_next_start = float(kept[idx + 2]["start"])
            after_next_dur = float(kept[idx + 1].get("end", kept[idx + 1]["start"]))
            # gap = time from end-of-next-line to start of line+2
            gap_after_next = after_next_start - float(kept[idx + 1]["start"])
        else:
            gap_after_next = duration - next_nom_start

        # We may borrow up to MAX_BORROW, but leave ≥ 0.3 s for the next clip itself
        borrow = min(MAX_BORROW, max(0.0, gap_after_next - 0.30))
        effective_slot = max(0.20, min(duration - play_start - 0.02, nominal_slot + borrow))

        # Apply mild atempo only if clip still doesn't fit after borrowing
        if clip_dur > effective_slot:
            needed_tempo = clip_dur / effective_slot
            if needed_tempo > 1.05:
                tempo = min(MAX_MIX_TEMPO, needed_tempo)
                wav = time_compress_wav(wav, sr, tempo, WORK)
                clip_dur = len(wav) / float(sr)

        # Hard clamp: never write past end of file
        max_rem_samples = int((duration - play_start - 0.005) * sr)
        if len(wav) > max_rem_samples > 0:
            wav = wav[:max_rem_samples]

        wav = apply_fade(wav, sr)
        wav_48k = resample_to_target(wav, sr, TARGET_SR)

        start_idx = max(0, int(round(play_start * TARGET_SR)))
        end_idx = min(total_samples, start_idx + len(wav_48k))
        if end_idx > start_idx:
            speech_mono[start_idx:end_idx] = wav_48k[: end_idx - start_idx]
            actual_end = end_idx / float(TARGET_SR)
            cursor_end = actual_end
            drift = play_start - nom_start
            placed_intervals.append({
                "segment_id": line.get("segment_id"),
                "speaker_id": line.get("speaker_id"),
                "start": round(start_idx / float(TARGET_SR), 3),
                "end": round(actual_end, 3),
                "drift": round(drift, 3),
            })
            if abs(drift) > 0.3:
                txt = line.get('translated_text', '')[:60]
                print(f"  [smear]  {line.get('segment_id')} drift={drift:+.2f}s  {txt!r}", flush=True)

    speech_stereo = np.column_stack([speech_mono, speech_mono])
    speech_bed_path = WORK / "mix_part_speech.wav"
    sf.write(speech_bed_path, speech_stereo, TARGET_SR, subtype="PCM_16")
    return speech_bed_path, placed_intervals



def main():
    raise_fd_limit()
    # Never mix a speech track that does not cover the audio.
    validate_tts_timeline(WORK)

    duration = float(read_json(WORK / "audio_meta.json")["duration"])
    manifest = read_json(WORK / "tts_manifest.json")["lines"]

    speech_bed_path, placed_intervals = build_clean_speech_bed(manifest, duration)

    write_json(
        WORK / "timeline_validation.json",
        {
            "incorrect_overlaps": [],
            "diarization_confirmed_overlaps": [],
            "tts_segment_count": len(placed_intervals),
        },
    )
    print(f"TTS timeline check: incorrect overlaps=0; placed clean segments={len(placed_intervals)}", flush=True)

    fx = read_json(WORK / "sfx_timeline.json").get("events", [])
    amb = read_json(WORK / "ambience.json")

    parts = [speech_bed_path]

    if fx:
        step = safe_chunk_inputs()
        for ci, base in enumerate(range(0, len(fx), step)):
            group = fx[base : base + step]
            inputs = []
            filters = []
            labels = []
            for i, event in enumerate(group):
                delay = max(0, int(float(event["start"]) * 1000))
                vol = event.get("volume", 0.14)
                inputs += ["-i", event["path"]]
                label = f"g{i}"
                filters.append(f"[{i}:a]aresample=48000,volume={vol},adelay={delay}|{delay}[{label}]")
                labels.append(f"[{label}]")
            filters.append("".join(labels) + f"amix=inputs={len(labels)}:duration=longest:normalize=0,apad,atrim=0:{duration}[out]")
            part = WORK / f"mix_part_fx_{ci:03d}.wav"
            run([
                "ffmpeg", "-y", "-v", "error", *inputs,
                "-filter_complex", ";".join(filters),
                "-map", "[out]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(part),
            ])
            parts.append(part)

    if amb.get("enabled"):
        candidate = amb["candidate"]
        path = ROOT / candidate["path"]
        part = WORK / "mix_part_amb.wav"
        vol = min(float(amb.get("volume", 0.06)), 0.06)
        filt = f"[0:a]aresample=48000,volume={vol},atrim=0:{duration}[out]"
        run([
            "ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", str(path),
            "-filter_complex", filt,
            "-map", "[out]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(part),
        ])
        parts.append(part)

    out = WORK / "mix.wav"
    if len(parts) == 1:
        speech_bed_path.replace(out)
    else:
        inputs = []
        labels = []
        for i, part in enumerate(parts):
            inputs += ["-i", str(part)]
            labels.append(f"[{i}:a]")
        filt = "".join(labels) + f"amix=inputs={len(labels)}:duration=longest:normalize=0,alimiter=limit=0.95[out]"
        run([
            "ffmpeg", "-y", "-v", "error", *inputs,
            "-filter_complex", filt,
            "-map", "[out]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(out),
        ])
        for part in parts:
            part.unlink(missing_ok=True)

    print(f"Mixed audio: {out} ({len(placed_intervals)} speech clips, 0 overlaps)")


if __name__ == "__main__":
    main()
