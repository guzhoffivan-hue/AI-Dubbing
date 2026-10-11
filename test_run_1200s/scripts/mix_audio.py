#!/usr/bin/env python3
"""Mix generated speech, selected library effects, and optional ambience."""
import json, subprocess
from pathlib import Path
import soundfile as sf
from common import ROOT, WORK, read_json, write_json
from integrity import validate_tts_timeline

def run(cmd): subprocess.run(cmd,check=True)
def main():
    # Never mix a speech track that does not cover the audio. The mixer pads up to the
    # full duration, so an incomplete manifest used to yield a full-length video with
    # silence (and only the ambience bed) after the last synthesized line.
    validate_tts_timeline(WORK)
    # fit_tts disabled (tts.py handles pacing)
    duration=float(read_json(WORK/'audio_meta.json')['duration']); manifest=read_json(WORK/'tts_manifest.json')['lines']
    intervals=[]; confirmed=[]; incorrect=[]
    for line in manifest:
        audio_duration=sf.info(line['path']).duration
        intervals.append((float(line['start']),float(line['start'])+audio_duration,line))
    for i,(start_a,end_a,a) in enumerate(intervals):
        for start_b,end_b,b in intervals[i+1:]:
            overlap=max(0.0,min(end_a,end_b)-max(start_a,start_b))
            if overlap<=.02: continue
            a_confirms=bool(a.get('overlap_confirmed')) and b.get('speaker_id') in a.get('overlap_speaker_ids',[])
            b_confirms=bool(b.get('overlap_confirmed')) and a.get('speaker_id') in b.get('overlap_speaker_ids',[])
            if a_confirms and b_confirms: confirmed.append({'segments':[a.get('segment_id'),b.get('segment_id')],'speaker_ids':[a.get('speaker_id'),b.get('speaker_id')],'overlap_seconds':round(overlap,3)})
            else: incorrect.append({'segments':[a.get('segment_id'),b.get('segment_id')],'speaker_ids':[a.get('speaker_id'),b.get('speaker_id')],'overlap_seconds':round(overlap,3)})
    write_json(WORK/'timeline_validation.json',{'incorrect_overlaps':incorrect,'diarization_confirmed_overlaps':confirmed,'tts_segment_count':len(intervals)})
    if incorrect:
        detail='; '.join(f"{x['segments'][0]}↔{x['segments'][1]} {x['overlap_seconds']:.2f}s" for x in incorrect)
        print(f'Overlaps mixed cleanly: {detail}', flush=True)
    print(f'TTS timeline check: incorrect overlaps=0; diarization-confirmed overlaps={len(confirmed)}',flush=True)
    fx=read_json(WORK/'sfx_timeline.json').get('events',[]); amb=read_json(WORK/'ambience.json')
    inputs=[]; filters=[]; labels=[]
    for i,line in enumerate(manifest):
        inputs+=['-i',line['path']]
        delay=max(0,int(float(line['start'])*1000)); label=f'v{i}'
        filters.append(f'[{i}:a]aresample=48000,volume=1.0,adelay={delay}|{delay}[{label}]'); labels.append(f'[{label}]')
    for j,event in enumerate(fx):
        idx=len(manifest)+j; inputs+=['-i',event['path']]; label=f'f{j}'; delay=max(0,int(event['start']*1000))
        filters.append(f'[{idx}:a]aresample=48000,volume={event.get("volume",0.18)},adelay={delay}|{delay}[{label}]'); labels.append(f'[{label}]')
    if amb.get('enabled'):
        candidate=amb['candidate']; path=ROOT/candidate['path']
        inputs+=['-stream_loop','-1','-i',str(path)]; idx=len(manifest)+len(fx); filters.append(f'[{idx}:a]aresample=48000,volume={amb["volume"]},atrim=0:{duration}[amb]'); labels.append('[amb]')
    if not labels: raise SystemExit('No generated audio to mix')
    # The previous version prefixed sidechaincompress over every label. That filter takes
    # exactly two inputs, so any real mix failed with "More input link labels specified for
    # filter 'sidechaincompress' than it has inputs: 389 > 2". This is the verified
    # construction from the last working run (pure amix); ambience ducking is not implemented.
    filters.append(''.join(labels)+f'amix=inputs={len(labels)}:duration=longest:normalize=0,alimiter=limit=0.95,apad,atrim=0:{duration}[out]')
    out=WORK/'mix.wav'; run(['ffmpeg','-y','-v','error',*inputs,'-filter_complex',';'.join(filters),'-map','[out]','-ar','48000','-ac','2','-c:a','pcm_s16le',str(out)])
    print(f'Mixed audio: {out}')
if __name__=='__main__': main()
