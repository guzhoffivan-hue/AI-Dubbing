
import shutil

def get_stretch_filter(tempo):
    # VideoLingo stretch: preserve formant and pitch
    if shutil.which('rubberband'):
        return f"rubberband=tempo={tempo:.4f}:pitch=1.0"
    return f"atempo={tempo:.4f}"

#!/usr/bin/env python3
"""Fit TTS lines into their slots: atempo up to 1.2x, then trim+fade. Idempotent (reads tts_manifest.orig.json)."""
import shutil, subprocess
import soundfile as sf
from common import WORK, read_json, write_json
MAXT=1.7; GAP=0.05
def main():
    mf=WORK/'tts_manifest.json'; bak=WORK/'tts_manifest.orig.json'
    if not bak.exists() or not any('fit_tempo' in l for l in read_json(mf)['lines']): shutil.copy(mf,bak)
    data=read_json(bak); lines=data['lines']; out=WORK/'tts_fit'; out.mkdir(exist_ok=True)
    order=sorted(range(len(lines)),key=lambda i:float(lines[i]['start'])); sped=trimmed=0
    for k,i in enumerate(order):
        l=lines[i]; s=float(l['start'])
        e=float(l['end']) if 'end' in l else s+float(l.get('target_duration_seconds',0) or 0)
        if k+1<len(order): e=min(e,float(lines[order[k+1]]['start'])-GAP)
        slot=max(e-s,0.3); dur=sf.info(l['path']).duration
        if dur<=slot+0.02: continue
        tempo=min(dur/slot,MAXT); af=get_stretch_filter(tempo)
        if dur/tempo>slot: af+=f',atrim=0:{slot:.3f},afade=t=out:st={max(slot-0.08,0):.3f}:d=0.08'; trimmed+=1
        dst=out/f'{i:04d}.wav'
        subprocess.run(['ffmpeg','-y','-v','error','-i',l['path'],'-af',af,'-c:a','pcm_s16le',str(dst)],check=True)
        l['path']=str(dst); l['fit_tempo']=round(tempo,3); sped+=1
        # Keep the manifest truthful: mix_audio.py validates the declared duration
        # against the file on disk, so a fitted file must publish its new duration.
        fitted=sf.info(str(dst)).duration
        l['tts_duration']=round(fitted,3); l['tts_end']=round(float(l['start'])+fitted,3)
        print(f"fit {l.get('segment_id',i)}: {dur:.2f}s -> slot {slot:.2f}s tempo={tempo:.2f}"+(' TRIMMED' if dur/tempo>slot else ''),flush=True)
    write_json(mf,data); print(f'fit_tts: sped={sped} (of which trimmed={trimmed}) of {len(lines)}',flush=True)
if __name__=='__main__': main()
