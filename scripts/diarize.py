#!/usr/bin/env python3
"""Run pyannote speaker diarization on Apple Silicon MPS + neural Wav2Vec2 Age/Gender recognition."""
import argparse
import gc
import os
import numpy as np
import torch
import torch.nn as nn
from common import WORK, config, write_json, load_env

DEFAULT_GENDER_MODEL = 'audeering/wav2vec2-large-robust-6-ft-age-gender'


class _ModelHead(nn.Module):
    def __init__(self, cfg, num_labels):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.final_dropout)
        self.out_proj = nn.Linear(cfg.hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


def _load_age_gender_model(model_name, device):
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.config = cfg
            self.wav2vec2 = Wav2Vec2Model(cfg)
            self.age = _ModelHead(cfg, 1)
            self.gender = _ModelHead(cfg, 3)
            self.init_weights()

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            hidden_states = torch.mean(outputs[0], dim=1)
            logits_age = self.age(hidden_states)
            probs_gender = torch.softmax(self.gender(hidden_states), dim=1)
            return hidden_states, logits_age, probs_gender

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = AgeGenderModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
    return processor, model


def _estimate_f0_stats(audio, sr, turns):
    import librosa
    f0_parts = []
    voiced_seconds = 0.0
    for turn in turns[:15]:
        clip = audio[max(0, int(turn['start'] * sr)):min(len(audio), int(turn['end'] * sr))]
        if len(clip) < int(0.6 * sr):
            continue
        if len(clip) > int(6.0 * sr):
            mid = len(clip) // 2
            clip = clip[mid - int(3.0 * sr):mid + int(3.0 * sr)]
        f0, _, prob = librosa.pyin(
            clip, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=256, fill_na=np.nan
        )
        good = f0[np.isfinite(f0) & (prob >= 0.55)]
        if len(good):
            f0_parts.extend(good.tolist())
            voiced_seconds += len(good) * 256 / sr
        if voiced_seconds >= 12.0:
            break
    if not f0_parts:
        return 0.0, 0.0, 0.0, 0.0
    values = np.asarray(f0_parts, dtype=float)
    return (
        float(np.median(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
        float(voiced_seconds),
    )


def classify_speakers_gender(audio_path, speakers, turns, overlap_intervals, cfg):
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    gender_model_id = cfg.get('gender_model', DEFAULT_GENDER_MODEL)

    processor = None
    model = None
    try:
        processor, model = _load_age_gender_model(gender_model_id, device)
    except Exception as exc:
        print(f'WARNING: Neural gender model unavailable ({exc}); falling back to pitch floor.', flush=True)

    def _is_clean_turn(t):
        for ov in overlap_intervals:
            if min(t['end'], ov['end']) - max(t['start'], ov['start']) > 0.15:
                return False
        return True

    voice_types = {}
    timbre_embeddings = {}

    for speaker_id in speakers:
        spk_turns = [t for t in turns if t['speaker_id'] == speaker_id]
        clean_turns = [t for t in spk_turns if _is_clean_turn(t)] or spk_turns
        turns_sorted = sorted(clean_turns, key=lambda t: t['end'] - t['start'], reverse=True)
        total_spk_dur = sum(max(0.0, t['end'] - t['start']) for t in spk_turns)

        median_f0, q25_f0, q75_f0, voiced_sec = _estimate_f0_stats(audio, sr, turns_sorted)

        if model is not None and processor is not None:
            windows = []
            weights = []
            for t in turns_sorted:
                c = audio[max(0, int(t['start'] * sr)):min(len(audio), int(t['end'] * sr))]
                if len(c) < int(0.35 * sr):
                    continue
                max_len = int(4.0 * sr)
                if len(c) > max_len:
                    mid = len(c) // 2
                    c = c[mid - max_len // 2:mid + max_len // 2]
                windows.append(c)
                weights.append(len(c) / float(sr))
                if len(windows) >= 8:
                    break
            if not windows and turns_sorted:
                t0 = turns_sorted[0]
                c = audio[max(0, int(t0['start'] * sr)):min(len(audio), int(t0['end'] * sr))]
                if len(c) > 0:
                    windows = [c[:int(4.0 * sr)]]
                    weights = [max(0.2, len(windows[0]) / float(sr))]

            g_list, a_list, emb_list = [], [], []
            for w in windows:
                if len(w) < int(0.5 * sr):
                    w = np.pad(w, (0, int(0.5 * sr) - len(w)))
                inp = processor(w, sampling_rate=sr, return_tensors='pt')['input_values'].to(device)
                with torch.no_grad():
                    emb, age, gender = model(inp)
                g_list.append(gender[0].detach().cpu().numpy())
                a_list.append(float(age[0, 0].detach().cpu()) * 100.0)
                emb_list.append(emb[0].detach().cpu().numpy())

            if g_list:
                w_arr = np.asarray(weights, dtype=float)
                w_arr = w_arr / np.sum(w_arr)
                probs = np.sum(np.asarray(g_list) * w_arr[:, None], axis=0)
                # id2label in audeering model: 0=female, 1=male, 2=child
                p_female = float(probs[0])
                p_male = float(probs[1])
                p_child = float(probs[2])
                age_years = round(float(np.median(a_list)), 1)
                mean_emb = np.mean(np.asarray(emb_list), axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 1e-8:
                    timbre_embeddings[speaker_id] = mean_emb / norm

                # Decide gender primarily by neural vocal-tract model; break borderline ties with F0 q25
                if abs(p_male - p_female) < 0.12 and q25_f0 > 0:
                    label = 'male' if q25_f0 < 165.0 else 'female'
                else:
                    label = 'male' if p_male >= p_female else 'female'

                margin = abs(p_male - p_female)
                confidence = 'high' if margin >= 0.35 else ('medium' if margin >= 0.15 else 'low')
                voice_types[speaker_id] = {
                    'voice_type': label,
                    'confidence': confidence,
                    'male_prob': round(p_male, 3),
                    'female_prob': round(p_female, 3),
                    'child_prob': round(p_child, 3),
                    'age_years': age_years,
                    'f0_median_hz': round(median_f0, 1) if median_f0 else None,
                    'f0_q25_hz': round(q25_f0, 1) if q25_f0 else None,
                    'f0_q75_hz': round(q75_f0, 1) if q75_f0 else None,
                    'voiced_seconds': round(max(voiced_sec, total_spk_dur), 2),
                    'method': f'neural Wav2Vec2 age-gender ({gender_model_id}) + acoustic F0 register',
                }
                continue

        # Fallback if neural model could not run: use baseline pitch floor (q25) resilient to Chinese tones
        if q25_f0 > 0:
            label = 'male' if (q25_f0 <= 162.0 or median_f0 <= 188.0) else 'female'
            confidence = 'medium' if voiced_sec >= 1.5 else 'low'
        else:
            label = 'male'
            confidence = 'low'
        voice_types[speaker_id] = {
            'voice_type': label,
            'confidence': confidence,
            'f0_median_hz': round(median_f0, 1) if median_f0 else None,
            'f0_q25_hz': round(q25_f0, 1) if q25_f0 else None,
            'f0_q75_hz': round(q75_f0, 1) if q75_f0 else None,
            'voiced_seconds': round(voiced_sec, 2),
            'method': 'pitch-floor q25 fallback heuristic',
        }

    if model is not None:
        del model, processor
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return voice_types, timbre_embeddings


def main():
    cfg = config()
    p = argparse.ArgumentParser()
    p.add_argument('--audio', default=str(WORK / 'audio.wav'))
    p.add_argument('--output', default=str(WORK / 'diarization.json'))
    a = p.parse_args()
    load_env()
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if not token:
        raise SystemExit('HF_TOKEN is required for the accepted gated pyannote model')

    model_id = cfg.get('diarization_model', 'pyannote/speaker-diarization-3.1')
    from pyannote.audio import Pipeline
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion, Specifications, Problem, Resolution])
    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)
    except TypeError as exc:
        if 'unexpected keyword argument' not in str(exc):
            raise
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)

    if torch.backends.mps.is_available():
        try:
            pipeline.to(torch.device('mps'))
        except Exception:
            pass

    min_speakers = int(cfg.get('diarization_min_speakers', 1))
    max_speakers = int(cfg.get('diarization_max_speakers', 8))

    try:
        raw = pipeline(a.audio, min_speakers=min_speakers, max_speakers=max_speakers, return_embeddings=True)
    except TypeError:
        raw = pipeline(a.audio, min_speakers=min_speakers, max_speakers=max_speakers)

    if isinstance(raw, tuple):
        raw, emb = raw
    else:
        emb = getattr(raw, 'speaker_embeddings', None)
    annotation = getattr(raw, 'speaker_diarization', raw)

    raw_turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        dur = float(turn.end) - float(turn.start)
        if dur < 0.08:
            continue
        raw_turns.append({'raw_speaker': str(speaker), 'start': float(turn.start), 'end': float(turn.end)})

    first_seen = {}
    for turn in raw_turns:
        first_seen[turn['raw_speaker']] = min(first_seen.get(turn['raw_speaker'], turn['start']), turn['start'])
    ordered = sorted(first_seen, key=first_seen.get)
    speaker_map = {raw_id: f'SPEAKER_{i:02d}' for i, raw_id in enumerate(ordered)}
    turns = sorted(
        [
            {
                'speaker_id': speaker_map[t['raw_speaker']],
                'start': t['start'],
                'end': t['end'],
                'confidence': 'not_calibrated_by_pyannote_3_1',
            }
            for t in raw_turns
        ],
        key=lambda x: (x['start'], x['end']),
    )
    speakers = [speaker_map[s] for s in ordered]

    # Compute total speaking time per speaker so smaller clusters always merge into dominant speakers
    spk_durations = {s: 0.0 for s in speakers}
    for t in turns:
        spk_durations[t['speaker_id']] = spk_durations.get(t['speaker_id'], 0.0) + max(0.0, t['end'] - t['start'])
    total_speech_dur = max(1e-3, sum(spk_durations.values()))

    sim = None
    merged = {}
    S_matrix = None
    sim_labels = []
    try:
        sim_labels = [speaker_map[l] for l in annotation.labels() if l in speaker_map]
        E = np.asarray(emb, dtype=float)
        E = E / np.maximum(1e-8, np.linalg.norm(E, axis=1, keepdims=True))
        S_matrix = E @ E.T
        sim = {'labels': sim_labels, 'cosine': np.round(S_matrix, 3).tolist()}

        thr = float(cfg.get('diarization_merge_cosine', 0.70) or 0.70)
        # Sort speakers from most dominant (longest speaking time) to shortest
        by_dominance = sorted(sim_labels, key=lambda s: (-spk_durations.get(s, 0.0), s))
        for idx_minor, spk_minor in enumerate(by_dominance):
            if spk_minor in merged:
                continue
            minor_dur = spk_durations.get(spk_minor, 0.0)
            is_tiny_phantom = (minor_dur < 5.0 and (minor_dur / total_speech_dur) < 0.03)
            best_target = None
            best_cos = -1.0
            for spk_major in by_dominance[:idx_minor]:
                target = merged.get(spk_major, spk_major)
                cos_val = float(S_matrix[sim_labels.index(spk_minor), sim_labels.index( target )])
                eff_thr = 0.52 if is_tiny_phantom else thr
                if cos_val >= eff_thr and cos_val > best_cos:
                    best_cos = cos_val
                    best_target = target
            if best_target is not None:
                merged[spk_minor] = best_target
    except Exception as exc:
        sim = {'unavailable': f'{type(exc).__name__}: {exc}'}

    if merged:
        for t in turns:
            while t['speaker_id'] in merged:
                t['speaker_id'] = merged[t['speaker_id']]
        speakers = [s for s in speakers if s not in merged]
        print('MERGED speakers (embedding similarity):', merged, flush=True)

    del pipeline, raw, annotation
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    overlap_intervals = []
    for i, left in enumerate(turns):
        for right in turns[i + 1:]:
            if right['start'] >= left['end']:
                break
            start = max(left['start'], right['start'])
            end = min(left['end'], right['end'])
            if end - start >= 0.05 and left['speaker_id'] != right['speaker_id']:
                overlap_intervals.append({'start': start, 'end': end, 'speaker_ids': [left['speaker_id'], right['speaker_id']]})

    voice_types, timbre_embeddings = classify_speakers_gender(a.audio, speakers, turns, overlap_intervals, cfg)

    # Secondary merge pass: if a minor speaker (<15% speech) has same neural gender,
    # high Wav2Vec2 vocal-tract similarity (>=0.965), and pyannote cosine >=0.60 with a major speaker
    if S_matrix is not None and len(speakers) > 1:
        spk_durations = {s: sum(max(0.0, t['end'] - t['start']) for t in turns if t['speaker_id'] == s) for s in speakers}
        by_dom = sorted(speakers, key=lambda s: (-spk_durations.get(s, 0.0), s))
        extra_merged = {}
        for idx_m, spk_m in enumerate(by_dom):
            if spk_m in extra_merged:
                continue
            if spk_durations.get(spk_m, 0.0) / total_speech_dur >= 0.18:
                continue
            for spk_dom in by_dom[:idx_m]:
                target = extra_merged.get(spk_dom, spk_dom)
                if voice_types.get(spk_m, {}).get('voice_type') != voice_types.get(target, {}).get('voice_type'):
                    continue
                if spk_m in sim_labels and target in sim_labels:
                    py_cos = float(S_matrix[sim_labels.index(spk_m), sim_labels.index(target)])
                else:
                    py_cos = 0.0
                w2v_cos = (
                    float(np.dot(timbre_embeddings[spk_m], timbre_embeddings[target]))
                    if (spk_m in timbre_embeddings and target in timbre_embeddings)
                    else 0.0
                )
                if py_cos >= 0.58 and w2v_cos >= 0.965:
                    extra_merged[spk_m] = target
                    break
        if extra_merged:
            merged.update(extra_merged)
            for t in turns:
                while t['speaker_id'] in extra_merged:
                    t['speaker_id'] = extra_merged[t['speaker_id']]
            speakers = [s for s in speakers if s not in extra_merged]
            voice_types = {s: v for s, v in voice_types.items() if s in speakers}
            print('MERGED speakers (timbre + gender consistency):', extra_merged, flush=True)

    uncertain = [t for t in turns if t['end'] - t['start'] < 0.75]
    result = {
        'model': model_id,
        'speaker_count_bounds': {
            'min_speakers': min_speakers,
            'max_speakers': max_speakers,
            'exact_count_forced': False,
        },
        'speaker_map': speaker_map,
        'speaker_ids': speakers,
        'speaker_count': len(speakers),
        'turns': turns,
        'overlap_intervals': overlap_intervals,
        'uncertain_turns': [
            {'speaker_id': t['speaker_id'], 'start': t['start'], 'end': t['end'], 'reason': 'shorter_than_0.75_seconds'}
            for t in uncertain
        ],
        'voice_types': voice_types,
        'speaker_similarity': sim,
        'merged_speakers': merged,
    }
    write_json(a.output, result)
    write_json(WORK / 'speakers.json', result)
    if len(speakers) < min_speakers:
        print(
            f'WARNING: diarization returned only {len(speakers)} speakers (minimum requested {min_speakers}); count constraint could not be met.',
            flush=True,
        )
    if uncertain:
        print(f'WARNING: {len(uncertain)} diarization turns are shorter than 0.75 seconds.', flush=True)
    print(
        f'Diarization model={model_id}; bounds={min_speakers}..{max_speakers}; detected={len(speakers)}; turns={len(turns)}; confirmed_overlap_intervals={len(overlap_intervals)}',
        flush=True,
    )
    for speaker_id in speakers:
        vt = voice_types.get(speaker_id, {})
        print(
            f'{speaker_id}: voice_type={vt.get("voice_type", "unknown")} '
            f'confidence={vt.get("confidence", "low")} '
            f'age={vt.get("age_years", "?")}y '
            f'male_prob={vt.get("male_prob", "?")}',
            flush=True,
        )


if __name__ == '__main__':
    main()
