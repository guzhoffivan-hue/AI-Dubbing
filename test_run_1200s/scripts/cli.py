#!/usr/bin/env python3
"""User-facing setup and resumable local dubbing command."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'logs'

def config():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('PyYAML is missing. Run ./setup to prepare the local Python environment.') from exc
    return yaml.safe_load((ROOT/'config.yaml').read_text()) or {}

def python_for(stage: str) -> Path:
    backend=config().get('tts_backend','qwen3') if stage=='tts' else ''
    env = 'translate' if stage=='tts' and backend=='qwen3_mlx' else {'asr':'asr','align':'asr','diarize':'asr','translate':'translate','tts':'tts'}.get(stage,'asr')
    p=ROOT/'.venvs'/env/'bin/python'
    if not p.exists():
        raise RuntimeError(f'Python environment for {stage} is missing ({p}). Run ./setup.')
    return p

def run_script(script: str, stage: str, work: Path, video: Path, output_file: Path|None=None, extra=()):
    env=os.environ.copy(); env.update(AI_DUB_WORKDIR=str(work),AI_DUB_INPUT_VIDEO=str(video),AI_DUB_OUTPUT_DIR=str(ROOT/'output'))
    if output_file: env['AI_DUB_OUTPUT_FILE']=str(output_file)
    cmd=[str(python_for(stage)),str(ROOT/'scripts'/script),*extra]
    logfile=LOGS/f'{work.name}_{stage}.log'; LOGS.mkdir(exist_ok=True)
    with logfile.open('a') as log:
        log.write('\n\n$ '+' '.join(cmd)+'\n'); log.flush()
        proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log.write(proc.stdout or ''); log.flush()
    if proc.returncode:
        detail=(proc.stdout or '').strip().splitlines()
        tail='\n'.join(detail[-12:])
        raise RuntimeError(f'{stage} failed. Details are in {logfile}.\n{tail}')

def probe_duration(video: Path) -> float:
    try:
        raw=subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(video)],text=True)
        return float(raw.strip())
    except Exception as exc: raise RuntimeError(f'Cannot read video metadata for {video.name}; check that FFmpeg supports this file.') from exc

def check_config(cfg):
    if cfg.get('use_original_audio', True) is not False:
        raise RuntimeError('Safety check blocked the run: use_original_audio must be exactly false in config.yaml.')

def step(work, status, key, description, action, total, done, started, durations):
    now=time.monotonic(); elapsed=now-started
    avg=sum(durations)/len(durations) if durations else 0
    remaining=max(0,avg*(total-done-1)) if avg else 0
    print(f'[{done+1}/{total}] {description} — elapsed {elapsed/60:.1f} min, estimated remaining {remaining/60:.1f} min',flush=True)
    status[key]='running'; write_status(work,status)
    t=time.monotonic()
    action()
    durations.append(time.monotonic()-t); status[key]='done'; write_status(work,status)

def write_status(work,status):
    p=work/'status.json'; p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n'); tmp.replace(p)

def process(video: Path, mode='full', test_duration=150, test_start=0, test_out=None):
    cfg=config(); check_config(cfg)
    dur=probe_duration(video); length=min(test_duration,max(0,dur-test_start)) if mode=='test' else dur
    if length<=0: raise RuntimeError('The selected test start is past the end of the video.')
    import re
    slug=re.sub(r'[^A-Za-z0-9._-]+','_',video.stem).strip('._') or 'video'
    work=ROOT/'work'/slug; work.mkdir(parents=True,exist_ok=True)
    outputs={'extract_audio':'audio.wav','asr':'asr.json','alignment':'alignment.json','diarization':'diarization.json','segment_merge':'segments.json','translation':'translated_segments.json','dialogue':'dialogue_segments.json','tts':'tts_manifest.json','sfx':'sfx_timeline.json','ambience':'ambience.json','mix':'mix.wav','mux':None}
    source_stat=video.stat(); signature={'pipeline_version':5,'tts_backend':cfg.get('tts_backend'),'tts_voice_mode':'one_profile_per_diarized_speaker_v2','diarization_model':cfg.get('diarization_model'),'diarization_min_speakers':cfg.get('diarization_min_speakers',1),'diarization_max_speakers':cfg.get('diarization_max_speakers',8),'translation_model':cfg.get('translation_model'),'path':str(video.resolve()),'size':source_stat.st_size,'mtime_ns':source_stat.st_mtime_ns,'mode':mode,'start':test_start if mode=='test' else 0,'duration':length}
    status_path=work/'status.json'; prior=json.loads(status_path.read_text()) if status_path.exists() else {}
    if prior.get('_signature') != signature: prior={}
    status={**{key:prior.get(key,'pending') for key in outputs},'_signature':signature}
    write_json_file(work/'audio_meta.json',{'duration':length,'start':test_start if mode=='test' else 0,'source':str(video.resolve())})
    write_status(work,status)
    phases=[('extract_audio','Extracting audio',lambda:run_script('extract_audio.py','asr',work,video,extra=('--duration',str(length),'--start',str(test_start if mode=='test' else 0),'--output',str(work/'audio.wav')))),
            ('asr','Transcribing',lambda:run_script('asr.py','asr',work,video)),
            ('alignment','Aligning',lambda:run_script('align.py','align',work,video)),
            ('diarization','Detecting speakers',lambda:run_script('diarize.py','diarize',work,video)),
            ('segment_merge','Building speaker-labelled segments',lambda:run_script('merge_segments.py','asr',work,video)),
            ('translation','Translating',lambda:run_script('translate.py','translate',work,video)),
            ('dialogue','Adapting dialogue timing',lambda:run_script('dialogue.py','asr',work,video)),
            ('tts','Generating Russian voices',lambda:run_script('tts.py','tts',work,video))]
    if cfg.get('enable_sfx',True): phases.append(('sfx','Building SFX',lambda:run_script('sfx.py','asr',work,video,extra=('timeline',))))
    else: phases.append(('sfx','Building SFX',lambda:write_json_file(work/'sfx_timeline.json',{'events':[]})))
    if cfg.get('enable_ambience',True): phases.append(('ambience','Selecting ambience',lambda:run_script('ambience.py','asr',work,video)))
    else: phases.append(('ambience','Selecting ambience',lambda:write_json_file(work/'ambience.json',{'enabled':False})))
    phases.extend([('mix','Mixing audio',lambda:run_script('mix_audio.py','asr',work,video)),
                   ('mux','Encoding output',lambda:run_script('mux.py','asr',work,video,test_out or ROOT/'output'/f'{video.stem}_ru.mp4'))])
    total=len(phases); started=time.monotonic(); durations=[]
    for key,label,action in phases:
        artifact=work/outputs[key] if outputs[key] else (test_out or ROOT/'output'/f'{video.stem}_ru.mp4')
        if status.get(key)=='done' and artifact and artifact.exists():
            print(f'[{sum(status.get(k)=="done" for k,_,_ in phases)}/{total}] {label} — already complete, resuming',flush=True); continue
        try: step(work,status,key,label,action,total,sum(status.get(k)=='done' for k,_,_ in phases),started,durations)
        except Exception as exc:
            status[key]='failed'; status['_error']=str(exc); write_status(work,status)
            friendly=str(exc)
            if key=='diarization' and ('HF_TOKEN' in friendly or 'token' in friendly.lower()):
                friendly += '\nAdd a Hugging Face token with access to pyannote/speaker-diarization-3.1 to .env (HF_TOKEN=...).'
            raise RuntimeError(f'Video: {video.name}\nStage: {label}\n{friendly}') from exc
    status['_elapsed_seconds']=round(time.monotonic()-started,1); write_status(work,status)
    print(f'Completed: {test_out or ROOT/"output"/f"{video.stem}_ru.mp4"}')

def write_json_file(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def setup():
    print('Checking local dubbing prerequisites…')
    if sys.platform!='darwin' or not __import__('platform').machine().startswith('arm'):
        raise RuntimeError('This setup targets Apple Silicon macOS.')
    for tool in ('ffmpeg','ffprobe'):
        if not shutil.which(tool):
            brew=shutil.which('brew')
            if not brew: raise RuntimeError(f'{tool} is missing. Install Homebrew, then rerun ./setup.')
            print('Installing FFmpeg with Homebrew…',flush=True); subprocess.run([brew,'install','ffmpeg'],check=True)
    cfg=config(); check_config(cfg)
    py311=shutil.which('python3.11')
    if not py311: raise RuntimeError('Python 3.11 is missing. Install it with `brew install python@3.11`, then rerun ./setup.')
    ensure_env('asr',py311,('torch','qwen_asr','pyannote'),('qwen-asr','pyannote.audio>=3.1,<4','soundfile','PyYAML'))
    ensure_env('translate',py311,('mlx_lm','mlx_audio','soundfile'),('mlx-lm','mlx-audio>=0.5.8','soundfile','PyYAML'))
    if cfg.get('tts_backend')=='qwen3': ensure_env('tts',py311,('torch','qwen_tts','soundfile'),('qwen-tts','soundfile','PyYAML'))
    sfx=ROOT/'work/sfx_index.json'
    if not sfx.exists():
        if not any((ROOT/'sfx_library').rglob('*.wav')): raise RuntimeError('Sonniss SFX library is missing or empty: sfx_library/.')
        print('Indexing local SFX library…',flush=True)
        run_script('sfx.py','asr',ROOT/'work'/'_setup',ROOT/'input'/'_placeholder.mp4',extra=('index','--output',str(sfx)))
    videos=sorted(p for p in (ROOT/'input').iterdir() if p.is_file() and p.suffix.lower() in {'.mp4','.mov','.mkv','.m4v','.avi'})
    if not videos:
        print('SETUP COMPLETE\nNo test video found in input/. Add a short video and run ./dub --test to validate the pipeline.'); return
    if not (os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN') or token_in_env()):
        raise RuntimeError('Packages and local SFX index are ready, but the bounded end-to-end test cannot run: pyannote requires a Hugging Face access token. Add HF_TOKEN=… to .env after accepting the model conditions, then run ./setup again. The 8 h 58 min video was not processed.')
    print('Running bounded 150-second setup test…',flush=True)
    process(videos[0],mode='test',test_duration=150,test_out=ROOT/'output'/'test_dub.mp4')
    test_output=ROOT/'output'/'test_dub.mp4'; start=(ROOT/'work'/slugify(videos[0].stem)/'audio.wav').stat().st_mtime; end=test_output.stat().st_mtime
    test_minutes=max(0,end-start)/60; ratio=probe_duration(videos[0])/150; estimate_hours=test_minutes*ratio/60
    print('SETUP COMPLETE\nTest output: '+str(test_output)+f'\nTest duration: 150 seconds (pipeline elapsed about {test_minutes:.1f} minutes)\nSFX indexed: '+str(json.loads(sfx.read_text()).get('audio_files','unknown'))+' files\nSelected TTS: '+cfg.get('tts_backend','not benchmarked')+' (MLX Qwen3-TTS 0.6B; runtime checked, comparison benchmark not run)'+f'\nEstimated full-video processing time: about {estimate_hours:.0f} hours by linear extrapolation\nRAM: per-process RSS observed up to about 4.4 GiB; total system peak was not measured\nProblems: no confident SFX/ambience was selected; scene-frame analysis and comparative TTS benchmark are not implemented.')

def ensure_env(name,py311,modules,packages):
    env=ROOT/'.venvs'/name; py=env/'bin'/'python'
    if not py.exists():
        print(f'Creating isolated Python environment: {name}',flush=True); subprocess.run([py311,'-m','venv',str(env)],check=True)
    check='import importlib.util,sys; sys.exit(any(importlib.util.find_spec(x) is None for x in '+repr(tuple(modules))+'))'
    if subprocess.run([str(py),'-c',check],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        print(f'Installing local dependencies for {name}…',flush=True)
        subprocess.run([str(py),'-m','pip','install','--upgrade','pip'],check=True)
        subprocess.run([str(py),'-m','pip','install',*packages],check=True)
    check_result=subprocess.run([str(py),'-c',check],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if check_result.returncode: raise RuntimeError(f'Could not prepare Python dependencies in .venvs/{name}. See pip output and rerun ./setup.')

def token_in_env():
    p=ROOT/'.env'
    if not p.exists(): return False
    return any(line.strip().startswith(('HF_TOKEN=','HUGGINGFACE_HUB_TOKEN=')) and line.partition('=')[2].strip().strip('"\'') for line in p.read_text().splitlines())

def slugify(name):
    import re
    return re.sub(r'[^A-Za-z0-9._-]+','_',name).strip('._') or 'video'

def main():
    ap=argparse.ArgumentParser(prog='dub',description='Local Russian video dubbing')
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('setup')
    d=sub.add_parser('dub'); d.add_argument('--test',action='store_true',help='process only a 150-second sample'); d.add_argument('--start',type=float,default=0); d.add_argument('--duration',type=float,default=150)
    a=ap.parse_args()
    try:
        if a.command=='setup': setup(); return 0
        cfg=config(); check_config(cfg)
        videos=sorted(p for p in (ROOT/'input').iterdir() if p.is_file() and p.suffix.lower() in {'.mp4','.mov','.mkv','.m4v','.avi'})
        if not videos: raise RuntimeError('No supported video files found in input/. Put a video there and run ./dub again.')
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'): raise RuntimeError('FFmpeg is missing. Run ./setup or install it with `brew install ffmpeg`.')
        for i,video in enumerate(videos,1):
            print(f'\nVideo {i}/{len(videos)}: {video.name}',flush=True)
            test_out=ROOT/'output'/'test_dub.mp4' if a.test and len(videos)==1 else None
            process(video,mode='test' if a.test else 'full',test_duration=a.duration,test_start=a.start,test_out=test_out)
        return 0
    except KeyboardInterrupt:
        print('\nStopped by user. Completed stages are saved and ./dub can resume.',file=sys.stderr); return 130
    except Exception as exc:
        print(f'\nERROR: {exc}',file=sys.stderr); return 1
if __name__=='__main__': raise SystemExit(main())
