#!/usr/bin/env python3
"""Run audio-only pyannote diarization independently of ASR and alignment."""
import argparse, gc, json, os, torch
from pathlib import Path
from common import ROOT, WORK, config, write_json, load_env

def main():
    cfg=config(); p=argparse.ArgumentParser()
    p.add_argument('--audio',default=str(WORK/'audio.wav'))
    p.add_argument('--output',default=str(WORK/'diarization.json'))
    a=p.parse_args(); load_env(); token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if not token: raise SystemExit('HF_TOKEN is required for the accepted gated pyannote model')
    model_id=cfg.get('diarization_model','pyannote/speaker-diarization-3.1')
    from pyannote.audio import Pipeline
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    if hasattr(torch.serialization,'add_safe_globals'):
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion, Specifications, Problem, Resolution])
    try: pipeline=Pipeline.from_pretrained(model_id,token=token)
    except TypeError as exc:
        if 'unexpected keyword argument' not in str(exc): raise
        pipeline=Pipeline.from_pretrained(model_id,use_auth_token=token)
    min_speakers=int(cfg.get('diarization_min_speakers',1)); max_speakers=int(cfg.get('diarization_max_speakers',8))
    # Pass a justified lower bound instead of forcing an exact speaker count.
    try: raw=pipeline(a.audio,min_speakers=min_speakers,max_speakers=max_speakers,return_embeddings=True)
    except TypeError: raw=pipeline(a.audio,min_speakers=min_speakers,max_speakers=max_speakers)
    if isinstance(raw,tuple): raw,emb=raw
    else: emb=getattr(raw,'speaker_embeddings',None)
    annotation=getattr(raw,'speaker_diarization',raw)
    raw_turns=[]
    for turn,_,speaker in annotation.itertracks(yield_label=True):
        raw_turns.append({'raw_speaker':str(speaker),'start':float(turn.start),'end':float(turn.end)})
    first_seen={}
    for turn in raw_turns: first_seen[turn['raw_speaker']]=min(first_seen.get(turn['raw_speaker'],turn['start']),turn['start'])
    ordered=sorted(first_seen,key=first_seen.get)
    speaker_map={raw_id:f'SPEAKER_{i:02d}' for i,raw_id in enumerate(ordered)}
    turns=sorted([{'speaker_id':speaker_map[t['raw_speaker']],'start':t['start'],'end':t['end'],'confidence':'not_calibrated_by_pyannote_3_1'} for t in raw_turns],key=lambda x:(x['start'],x['end']))
    speakers=[speaker_map[s] for s in ordered]
    sim=None; merged={}
    try:
        import numpy as np
        labels=[speaker_map[l] for l in annotation.labels()]; E=np.asarray(emb,dtype=float)
        E=E/np.linalg.norm(E,axis=1,keepdims=True); S=E@E.T
        sim={'labels':labels,'cosine':np.round(S,3).tolist()}
        thr=cfg.get('diarization_merge_cosine')
        if thr is not None:
            for i,la in enumerate(sorted(labels)):
                for lb in sorted(labels)[:i]:
                    if lb not in merged and S[labels.index(la),labels.index(lb)]>=float(thr): merged[la]=lb; break
    except Exception as exc: sim={'unavailable':f'{type(exc).__name__}: {exc}'}
    if merged:
        for t in turns: t['speaker_id']=merged.get(t['speaker_id'],t['speaker_id'])
        speakers=[s for s in speakers if s not in merged]; print('MERGED speakers:',merged,flush=True)
    overlap_intervals=[]
    for i,left in enumerate(turns):
        for right in turns[i+1:]:
            start=max(left['start'],right['start']); end=min(left['end'],right['end'])
            if end-start>=.05 and left['speaker_id']!=right['speaker_id']:
                overlap_intervals.append({'start':start,'end':end,'speaker_ids':[left['speaker_id'],right['speaker_id']]})
    voice_types={}
    try:
        import librosa, numpy as np
        audio,sr=librosa.load(a.audio,sr=16000,mono=True)
        for speaker_id in speakers:
            f0_parts=[]; voiced_seconds=0.0
            for turn in [t for t in turns if t['speaker_id']==speaker_id]:
                clip=audio[max(0,int(turn['start']*sr)):min(len(audio),int(turn['end']*sr))]
                if len(clip)<int(.8*sr): continue
                f0,_,prob=librosa.pyin(clip,fmin=60,fmax=400,sr=sr,frame_length=1024,hop_length=256,fill_na=np.nan)
                good=f0[np.isfinite(f0)&(prob>=.55)]
                if len(good): f0_parts.extend(good.tolist()); voiced_seconds+=len(good)*256/sr
            if f0_parts:
                values=np.asarray(f0_parts); median=float(np.median(values)); q25=float(np.percentile(values,25)); q75=float(np.percentile(values,75))
                female_share=float(np.mean(values>=165)); male_share=float(np.mean(values<=165))
            else: median=q25=q75=0.0; female_share=male_share=0.0
            if voiced_seconds>=1.5 and median>=185 and q25>=165 and female_share>=.7: label='female'; confidence='high' if voiced_seconds>=3 else 'medium'
            elif voiced_seconds>=1.5 and median<=150 and q75<=175 and male_share>=.7: label='male'; confidence='high' if voiced_seconds>=3 else 'medium'
            else: label='unknown'; confidence='low'
            voice_types[speaker_id]={'voice_type':label,'confidence':confidence,'f0_median_hz':round(median,1) if median else None,'f0_q25_hz':round(q25,1) if q25 else None,'f0_q75_hz':round(q75,1) if q75 else None,'voiced_seconds':round(voiced_seconds,2),'method':'conservative pYIN acoustic heuristic; ambiguous or short samples are unknown'}
    except Exception as exc:
        voice_types={s:{'voice_type':'unknown','confidence':'low','method':f'acoustic F0 estimate unavailable: {type(exc).__name__}'} for s in speakers}
    uncertain=[t for t in turns if t['end']-t['start']<.75]
    result={'model':model_id,'speaker_count_bounds':{'min_speakers':min_speakers,'max_speakers':max_speakers,'exact_count_forced':False},'speaker_map':speaker_map,'speaker_ids':speakers,'speaker_count':len(speakers),'turns':turns,'overlap_intervals':overlap_intervals,'uncertain_turns':[{'speaker_id':t['speaker_id'],'start':t['start'],'end':t['end'],'reason':'shorter_than_0.75_seconds'} for t in uncertain],'voice_types':voice_types,'speaker_similarity':sim,'merged_speakers':merged}
    del pipeline, raw, annotation; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    write_json(a.output,result); write_json(WORK/'speakers.json',result)
    if len(speakers)<min_speakers: print(f'WARNING: diarization returned only {len(speakers)} speakers (minimum requested {min_speakers}); count constraint could not be met.',flush=True)
    if uncertain: print(f'WARNING: {len(uncertain)} diarization turns are shorter than 0.75 seconds.',flush=True)
    print(f'Diarization model={model_id}; bounds={min_speakers}..{max_speakers}; detected={len(speakers)}; turns={len(turns)}; confirmed_overlap_intervals={len(overlap_intervals)}',flush=True)
    for speaker_id in speakers: print(f'{speaker_id}: voice_type={voice_types.get(speaker_id,{}).get("voice_type","unknown")} confidence={voice_types.get(speaker_id,{}).get("confidence","low")}',flush=True)
if __name__=='__main__': main()
