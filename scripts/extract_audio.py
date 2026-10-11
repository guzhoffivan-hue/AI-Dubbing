#!/usr/bin/env python3
"""Extract a short mono 16 kHz work segment; never used as final audio."""
import argparse, subprocess
from pathlib import Path
from common import ROOT, WORK, INPUT_VIDEO, config

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path); ap.add_argument('--duration',type=float); ap.add_argument('--start',type=float,default=0); ap.add_argument('--output',type=Path,default=WORK/'audio.wav'); a=ap.parse_args()
    cfg=config(); source=a.input or INPUT_VIDEO or next((ROOT/'input').glob('*.mp4'),None)
    if source is None or not source.exists(): raise SystemExit('No MP4 found under input/')
    duration=a.duration or cfg.get('test_duration_seconds',150)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(a.start),'-i',str(source),'-t',str(duration),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(a.output)]
    subprocess.run(cmd,check=True)
    print(f'Extracted {duration:.1f}s source audio to {a.output}')
if __name__=='__main__': main()
