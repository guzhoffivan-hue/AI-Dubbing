#!/usr/bin/env python3
"""Mux video frames with only the newly generated Russian audio."""
import subprocess
from pathlib import Path
from common import ROOT, WORK, INPUT_VIDEO, OUTPUT_DIR, config

def main():
    if config().get('use_original_audio',True): raise SystemExit('Safety check: original audio must remain disabled')
    src=INPUT_VIDEO or next((ROOT/'input').glob('*.mp4')); audio=WORK/'mix.wav'; out=Path(__import__('os').environ.get('AI_DUB_OUTPUT_FILE', OUTPUT_DIR/'test_dub.mp4')); out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(src),'-i',str(audio),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',str(out)],check=True)
    print(f'Created {out}')
if __name__=='__main__': main()
