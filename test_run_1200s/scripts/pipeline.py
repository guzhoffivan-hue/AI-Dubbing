#!/usr/bin/env python3
"""Run a bounded local dubbing test; full-length processing needs explicit opt-in."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from common import ROOT, config, write_json, load_env

def run(script, env, *args):
    python=ROOT/'.venvs'/env/'bin/python'
    cmd=[str(python),str(ROOT/'scripts'/script),*args]
    print('\n>>> '+script,flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True,env=os.environ.copy())

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    t=sub.add_parser('test'); t.add_argument('--duration',type=float,default=None); t.add_argument('--start',type=float,default=0)
    sub.add_parser('resume'); full=sub.add_parser('full'); full.add_argument('--confirm-full-length',action='store_true')
    a=p.parse_args(); load_env(); cfg=config()
    if cfg.get('use_original_audio',True): raise SystemExit('Safety check failed: config.yaml must keep use_original_audio: false')
    source=next((ROOT/'input').glob('*.mp4'),None)
    if source is None: raise SystemExit('No input MP4 found')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(source)],text=True))
    total=float(probe['format']['duration'])
    if a.mode=='full' and not a.confirm_full_length: raise SystemExit('Full run blocked. User requested only a short test; do not confirm without a separate explicit request.')
    duration=(a.duration or cfg.get('test_duration_seconds',150)) if a.mode!='full' else total
    start=0 if a.mode=='full' else a.start
    # if duration>300 and a.mode!='full': raise SystemExit('Test duration is limited to 300 seconds')
    # if a.mode=='full' and total>300: raise SystemExit('Full processing has not been explicitly authorized; use test or resume')
    t0=time.monotonic()
    subprocess.run([sys.executable,str(ROOT/'scripts/extract_audio.py'),'--input',str(source),'--duration',str(duration),'--start',str(start)],cwd=ROOT,check=True)
    write_json(ROOT/'work/audio_meta.json',{'duration':duration,'start':start,'source':str(source)})
    run('asr.py','asr'); run('align.py','asr'); run('diarize.py','asr'); run('translate.py','translate'); run('dialogue.py','tts'); run('tts.py','tts')
    if cfg.get('enable_sfx'): run('sfx.py','asr','timeline')
    else: write_json(ROOT/'work/sfx_timeline.json',{'events':[]})
    if cfg.get('enable_ambience'): run('ambience.py','asr')
    else: write_json(ROOT/'work/ambience.json',{'enabled':False})
    run('mix_audio.py','asr'); run('mux.py','asr')
    elapsed=time.monotonic()-t0
    write_json(ROOT/'work/pipeline_stats.json',{'duration':duration,'start':start,'elapsed_seconds':elapsed,'source_duration_seconds':total})
    print(f'Pipeline complete in {elapsed:.1f}s')
if __name__=='__main__': main()
