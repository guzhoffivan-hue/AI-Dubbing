#!/usr/bin/env python3
"""Универсальная диагностика: спикеры, голоса, пол по репликам, реальная скорость TTS. Берёт самую свежую work-папку."""
import json, sys, collections, pathlib, numpy as np
import soundfile as sf, librosa
root=pathlib.Path(__file__).resolve().parent.parent
W=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else max([p for p in (root/'work').iterdir() if p.is_dir() and (p/'translated_segments.json').exists()],key=lambda p:p.stat().st_mtime)
print('work:',W.name)
L=json.load(open(W/'translated_segments.json'))['lines']; dur=collections.defaultdict(float); cnt=collections.Counter()
y,sr=librosa.load(str(W/'audio.wav'),sr=16000); f0s=collections.defaultdict(list)
for l in L:
    s=l['speaker_id']; cnt[s]+=1; dur[s]+=l['end']-l['start']
    seg=y[int(l['start']*sr):int(l['end']*sr)]
    if len(seg)>=sr*0.8:
        f,_,_=librosa.pyin(seg,fmin=70,fmax=400,sr=sr); v=f[~np.isnan(f)]
        if len(v)>10: f0s[s].append(float(np.median(v)))
tot=sum(dur.values())
for s in sorted(cnt):
    a=np.array(f0s[s]) if f0s[s] else np.array([np.nan])
    print(f"{s}: {cnt[s]} реплик, {dur[s]:.0f}s ({100*dur[s]/tot:.0f}% речи), f0 по репликам: медиана={np.nanmedian(a):.0f}Hz IQR={np.nanpercentile(a,25):.0f}-{np.nanpercentile(a,75):.0f} n={len(f0s[s])}")
vp=json.load(open(W/'voice_profiles.json')); print('voice mapping:',vp['speaker_to_voice'],{k:v['voice_type'] for k,v in vp['profiles'].items()})
try:
    M=json.load(open(W/'tts_manifest.json'))['lines']; by={l['segment_id']:l for l in L}; r=[]
    for m in M:
        t=by[m['segment_id']]['translated_text']; d=sf.info(m['path']).duration
        if d>0.8: r.append(len(t)/d)
    print(f"реальная скорость TTS: медиана={np.median(r):.1f} симв/с, p10={np.percentile(r,10):.1f}, p90={np.percentile(r,90):.1f} (n={len(r)})")
except Exception as e: print('tts calibration skipped:',e)
