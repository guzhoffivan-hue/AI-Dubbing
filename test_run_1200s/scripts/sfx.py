#!/usr/bin/env python3
"""Offline Sonniss library scanner and filename-based sound search."""
from __future__ import annotations
import argparse, json, re, subprocess, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('AI_DUB_WORKDIR', ROOT / 'work')).resolve()
AUDIO_EXTS = {'.wav', '.wave', '.aif', '.aiff', '.flac', '.mp3', '.m4a', '.ogg', '.opus', '.caf'}
RULES = {
 'footsteps':['footstep','steps','walking','walk','run','running','flip flop'],
 'doors':['door','hinge','lock','latch'], 'clothing':['cloth','clothing','fabric','shirt','jacket','velcro'],
 'objects':['drop','pickup','handling','object','case','bag'], 'kitchen':['kitchen','cup','glass clink','plate','cutlery','cooking'],
 'phones':['phone','telephone','ringtone','mobile'], 'electronics':['electronic','electric','machine','alarm','ui','interface','radio','computer'],
 'vehicles':['vehicle','car','truck','train','motorcycle','engine','pass-by','traffic','commute'],
 'impacts':['impact','hit','punch','slam','bang','boom','crash','collision'], 'water':['water','rain','liquid','drip','splash','river','pool'],
 'paper':['paper','book','page','card','scratch card'], 'wood':['wood','wooden','timber'], 'metal':['metal','iron','steel','metallic'],
 'glass':['glass','crystal'], 'nature':['forest','bird','animal','nature','wind','leaves','cricket','campfire'],
 'weather':['weather','storm','thunder','wind','rain','fireworks'], 'room_tone':['room tone','roomtone','interior ambience','fridge hum','office'],
 'street':['street','urban','city','traffic','public spaces'], 'crowd':['crowd','walla','voices','audience','match','restaurant','bar'],
 'movement':['movement','foley','climb','run','swing','swoosh','whoosh','rush','pass by'], 'ambience':['ambience','ambient','atmosphere','environment','interior','exterior'],
 'weapons':['weapon','sword','gun','melee','rifle','blade'], 'creatures':['creature','monster','dinosaur','beast','growl'],
 'music_like':['musical','music box','drone','saxophone','horn'], 'designed':['designed','cinematic','transition','sci-fi','magic','glitch']
}
def classify(path: Path):
    text = re.sub(r'[_-]+', ' ', (path.parent.name + ' ' + path.stem).lower())
    cats = [c for c, words in RULES.items() if any(w in text for w in words)]
    if not cats: cats = ['uncategorized']
    tags = sorted(set(re.findall(r'[a-z0-9]+', text)))
    return cats, tags

def probe(path: Path):
    cmd=['ffprobe','-v','error','-select_streams','a:0','-show_entries','stream=codec_name,sample_rate,channels,channel_layout,bits_per_sample:format=duration,size','-of','json',str(path)]
    try:
        data=json.loads(subprocess.check_output(cmd,stderr=subprocess.DEVNULL,text=True))
        stream=(data.get('streams') or [{}])[0]; fmt=data.get('format',{})
        duration=float(fmt['duration']) if fmt.get('duration') else None
        return {'duration':duration,'sample_rate':int(stream['sample_rate']) if stream.get('sample_rate') else None,
                'channels':stream.get('channels'),'channel_layout':stream.get('channel_layout'),
                'codec':stream.get('codec_name'),'bits_per_sample':stream.get('bits_per_sample'),
                'size_bytes':int(fmt['size']) if fmt.get('size') else path.stat().st_size}
    except Exception as e:
        return {'duration':None,'sample_rate':None,'channels':None,'channel_layout':None,'codec':None,'bits_per_sample':None,'size_bytes':path.stat().st_size,'probe_error':str(e)}

def build_index(root: Path, output: Path):
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    rows=[]; failures=0
    for i,p in enumerate(files,1):
        cats,tags=classify(p); info=probe(p)
        if info.get('probe_error'): failures+=1
        rows.append({'path':p.relative_to(ROOT).as_posix() if str(p).startswith(str(ROOT) + '/') else str(p), 'filename':p.name,'category':cats,'tags':tags,'loudness_lufs':None,**info})
        if i%50==0: print(f'Indexed {i}/{len(files)}',flush=True)
    result={'schema_version':1,'library_root':str(root),'audio_files':len(rows),'probe_failures':failures,
            'categories':sorted({c for x in rows for c in x['category']}),'loudness_note':'Not measured during metadata scan; null means unavailable.', 'files':rows}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'Indexed {len(rows)} files; {failures} probe failures; {len(result["categories"])} categories → {output}')

def make_timeline():
    from common import read_json, write_json
    index=json.loads((ROOT/'work'/'sfx_index.json').read_text()); lines=read_json(WORK/'translation.json')['lines']
    cues={'footsteps':['walk','walking','run','running','footstep','steps','идет','идут','бежит','шаг'],
          'doors':['door','opens','closes','двер','открой','закрой'], 'phones':['phone','call','ring','телефон','звонит'],
          'glass':['glass','bottle','cup','стакан','бутыл','чаш'], 'impacts':['hit',' удар',' ударил','падает','crash','bang'],
          'water':['rain','water','splash','дожд','вода','река'], 'vehicles':['car','truck','engine','машин','автомоб','двигатель']}
    used=[]; previous=-99
    for line in lines:
        if line['start']-previous<5: continue
        text=(' '+line.get('translated_text','').lower()+' '+line.get('source_text','').lower())
        category=next((c for c,words in cues.items() if any(w in text for w in words)),None)
        if not category: continue
        pool=[x for x in index['files'] if category in x['category'] and x.get('duration') and x['duration']>0.5]
        if not pool: continue
        clip=pool[0]
        used.append({'start':float(line['start']),'duration':min(clip['duration'],3.0),'path':str(ROOT/clip['path']),'category':category,'volume':0.14,'matched_text':line.get('translated_text','')})
        previous=line['start']
        if len(used)>=6: break
    write_json(WORK/'sfx_timeline.json',{'events':used,'note':'Conservative keyword-to-filename match against local library.'})
    print(f'Selected {len(used)} sound effects from local library')

def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='cmd',required=True)
    scan=sub.add_parser('index'); scan.add_argument('--library',type=Path,default=ROOT/'sfx_library'); scan.add_argument('--output',type=Path,default=WORK/'sfx_index.json')
    search=sub.add_parser('search'); search.add_argument('query'); search.add_argument('--index',type=Path,default=WORK/'sfx_index.json'); search.add_argument('--limit',type=int,default=10)
    sub.add_parser('timeline')
    a=ap.parse_args()
    if a.cmd=='index': build_index(a.library,a.output)
    elif a.cmd=='timeline': make_timeline()
    else:
        data=json.loads(a.index.read_text()); terms=set(re.findall(r'[a-z0-9]+',a.query.lower())); scored=[]
        for x in data['files']:
            hay=set(x['tags'])|set(x['category']); score=len(terms&hay)
            if score: scored.append((score,x))
        for _,x in sorted(scored,key=lambda p:(-p[0],p[1]['filename']))[:a.limit]: print(f"{x['category']} | {x['duration']}s | {x['path']}")
if __name__=='__main__': main()
