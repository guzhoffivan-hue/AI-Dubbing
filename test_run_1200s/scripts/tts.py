#!/usr/bin/env python3
"""Generate dialogue with one persistent Russian voice profile per speaker."""
import gc, hashlib, json, subprocess
from collections import Counter
from pathlib import Path
import soundfile as sf
from common import ROOT, WORK, config, read_json, write_json

PROFILE_TEXT='Сегодня тихо светает. Я расскажу эту историю спокойно, ясно и выразительно.'
PROFILE_VERSION=1
DESIGN_MODEL='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16'
RATES=[]
CLONE_MODEL='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16'

def voice_instruction(voice_type, ordinal):
    traits=['warm, clear, steady timbre','light, bright, articulate timbre','rounded, resonant, composed timbre','slightly textured, calm, expressive timbre']
    trait=traits[ordinal % len(traits)]
    if voice_type=='female':
        identity='An adult female native Russian speaker with a naturally feminine mid-to-high register'
    elif voice_type=='male':
        identity='An adult male native Russian speaker with a naturally masculine mid register'
    else:
        identity='An adult native Russian speaker with a natural, neutral middle register; do not force a strongly masculine or feminine sound'
    return f'{identity}, {trait}. Standard contemporary Russian pronunciation, clear Moscow standard accent, no Chinese accent. Natural narration with measured, expressive intonation.'

def build_voice_profiles(speaker_ids, voice_types, segment_counts):
    """Design each voice once, persist it, then reuse its same audio as clone reference."""
    profile_dir=WORK/'voice_profiles'; profile_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=WORK/'voice_profiles.json'
    saved=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    saved_ok=(saved.get('profile_version')==PROFILE_VERSION and saved.get('model')==DESIGN_MODEL and set(saved.get('speaker_to_voice',{}))==set(speaker_ids))
    speaker_to_voice=saved.get('speaker_to_voice',{}) if saved_ok else {}
    profiles=saved.get('profiles',{}) if saved_ok else {}
    missing=[]
    for ordinal,speaker_id in enumerate(speaker_ids):
        voice_id=speaker_to_voice.get(speaker_id, f'VOICE_{ordinal:02d}')
        speaker_to_voice[speaker_id]=voice_id
        profile=profiles.get(voice_id,{})
        ref=Path(profile.get('reference_audio','')) if profile.get('reference_audio') else None
        expected_type=voice_types.get(speaker_id,{}).get('voice_type','unknown')
        expected_instruction=voice_instruction(expected_type,ordinal)
        if not profile.get('reference_text')==PROFILE_TEXT or profile.get('voice_type')!=expected_type or profile.get('instruction')!=expected_instruction or not ref or not ref.exists() or ref.stat().st_size<=1024:
            missing.append((ordinal,speaker_id,voice_id,expected_type,expected_instruction))
    if missing:
        from mlx_audio.tts.utils import load_model
        import numpy as np
        design_model=load_model(DESIGN_MODEL)
        for ordinal,speaker_id,voice_id,voice_type,instruction in missing:
            ref=profile_dir/f'{voice_id.lower()}.wav'
            results=list(design_model.generate_voice_design(text=PROFILE_TEXT,language='Russian',instruct=instruction))
            if not results: raise RuntimeError(f'VoiceDesign returned no reference audio for {speaker_id}')
            audio=np.asarray(results[0].audio).squeeze(); sr=int(results[0].sample_rate)
            sf.write(ref,audio,sr)
            profiles[voice_id]={'speaker_id':speaker_id,'voice_type':voice_type,'voice_confidence':voice_types.get(speaker_id,{}).get('confidence','low'),'instruction':instruction,'reference_text':PROFILE_TEXT,'reference_audio':str(ref),'reference_sample_rate':sr,'source':'one-time native Russian VoiceDesign sample; source audio is not used as a TTS clone reference'}
            write_json(manifest_path,{'profile_version':PROFILE_VERSION,'backend':'qwen3_mlx','model':DESIGN_MODEL,'clone_model':CLONE_MODEL,'speaker_to_voice':speaker_to_voice,'profiles':profiles})
        del design_model; gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception: pass
    voice_types_report={s:voice_types.get(s,{'voice_type':'unknown','confidence':'low'}) for s in speaker_ids}
    report={'profile_version':PROFILE_VERSION,'backend':'qwen3_mlx','model':DESIGN_MODEL,'clone_model':CLONE_MODEL,'speaker_to_voice':speaker_to_voice,'profiles':profiles,'voice_types':voice_types_report,'segment_counts':dict(segment_counts),'fallbacks':{s:'neutral unknown voice design' for s in speaker_ids if voice_types_report[s].get('voice_type')=='unknown'}}
    write_json(manifest_path,report)
    print('Speaker voice assignment:')
    for speaker_id in speaker_ids:
        info=voice_types_report[speaker_id]
        print(f'{speaker_id} | {info.get("voice_type","unknown")} | {speaker_to_voice[speaker_id]} | {segment_counts.get(speaker_id,0)} segments')
    return speaker_to_voice,profiles

def main():
    cfg=config(); backend=cfg.get('tts_backend','qwen3')
    data=read_json(WORK/'dialogue.json'); speaker_data=read_json(WORK/'speakers.json')
    lines=data['lines']
    speaker_ids=sorted({line.get('speaker_id',line.get('speaker','UNKNOWN')) for line in lines})
    segment_counts=Counter(line.get('speaker_id',line.get('speaker','UNKNOWN')) for line in lines)
    voice_types=speaker_data.get('voice_types',{})
    refs={}

    if backend=='qwen3_mlx':
        speaker_to_voice,profiles=build_voice_profiles(speaker_ids,voice_types,segment_counts)
        for line in lines:
            speaker_id=line.get('speaker_id',line.get('speaker','UNKNOWN'))
            line['speaker_id']=speaker_id; line['speaker']=speaker_id; line['voice_id']=speaker_to_voice[speaker_id]
            profile=profiles[line['voice_id']]
            refs[speaker_id]={'path':profile['reference_audio'],'text':profile['reference_text']}
        # Keep the explicit map in every dialogue segment for inspection and resume.
        write_json(WORK/'dialogue.json',{'lines':lines})
        write_json(WORK/'dialogue_segments.json',{'lines':lines,'timing_policy':'source start preserved; non-overlapping TTS is fitted before the next line'})
        write_json(WORK/'voice_mapping.json',{'speaker_to_voice':speaker_to_voice,'voice_types':{s:voice_types.get(s,{'voice_type':'unknown','confidence':'low'}) for s in speaker_ids},'profiles':profiles,'created_once_for_video':True})
        from mlx_audio.tts.utils import load_model
        import numpy as np
        import mlx.core as mx
        model=load_model(CLONE_MODEL)
    else:
        # Legacy backends retain their existing source-reference behavior, but
        # select that reference once per speaker and never once per segment.
        source=WORK/'audio.wav'; speakers=speaker_data
        for speaker_id in speaker_ids:
            utterances=[x for x in speakers['utterances'] if x.get('speaker_id',x.get('speaker'))==speaker_id]
            turns=[x for x in utterances if 2.5<=x['end']-x['start']<=12] or utterances
            if not turns: continue
            chosen=max(turns,key=lambda x:min(x['end']-x['start'],8)); p=WORK/f'reference_{speaker_id}.wav'
            subprocess.run(['ffmpeg','-y','-v','error','-ss',str(max(0,chosen['start'])),'-t',str(min(12,chosen['end']-chosen['start'])),'-i',str(source),'-ac','1','-ar','24000',str(p)],check=True)
            refs[speaker_id]={'path':str(p),'text':chosen['source_text']}
        if not refs: raise SystemExit('No speaker reference audio available')
        speaker_to_voice={speaker_id:f'VOICE_{i:02d}' for i,speaker_id in enumerate(speaker_ids)}
        for line in lines:
            speaker_id=line.get('speaker_id',line.get('speaker','UNKNOWN'))
            line['speaker_id']=speaker_id; line['speaker']=speaker_id; line['voice_id']=speaker_to_voice[speaker_id]
        write_json(WORK/'dialogue.json',{'lines':lines})
        if backend=='qwen3':
            import torch
            from qwen_tts import Qwen3TTSModel
            model=Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-Base',device_map='mps',dtype=torch.bfloat16)
        elif backend=='chatterbox':
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model=ChatterboxMultilingualTTS.from_pretrained(device='mps')
        elif backend=='cosyvoice3':
            raise RuntimeError('CosyVoice3 backend requires a local Fun-CosyVoice3 model installation.')
        else: raise RuntimeError(f'Unknown TTS backend: {backend}')
        profiles={voice_id:{'reference_audio':refs[speaker_id]['path']} for speaker_id,voice_id in speaker_to_voice.items()}
        print('Speaker voice assignment:')
        for speaker_id in speaker_ids:
            info=voice_types.get(speaker_id,{'voice_type':'unknown'})
            print(f'{speaker_id} | {info.get("voice_type","unknown")} | {speaker_to_voice[speaker_id]} | {segment_counts.get(speaker_id,0)} segments')

    outdir=WORK/'tts'; outdir.mkdir(parents=True,exist_ok=True); manifest=[]
    resume_path=WORK/'tts_resume.json'; resume=json.loads(resume_path.read_text()) if resume_path.exists() else {}
    profile_hash={voice_id:hashlib.sha256(Path(profile['reference_audio']).read_bytes()).hexdigest() for voice_id,profile in profiles.items()}
    for i,line in enumerate(lines):
        speaker_id=line.get('speaker_id',line.get('speaker','UNKNOWN')); voice_id=speaker_to_voice[speaker_id]; ref=refs[speaker_id]
        path=outdir/f'{i:04d}.wav'
        digest=hashlib.sha256((line['translated_text']+'\0'+voice_id+'\0'+profile_hash[voice_id]+'\0'+str(line.get('available_duration'))+'\0'+backend+'_fixed_profile_v2').encode()).hexdigest()
        if path.exists() and path.stat().st_size>1024 and resume.get(str(i),{}).get('text_sha256')==digest:
            info=sf.info(path); manifest.append({'i':i,'segment_id':line['segment_id'],'speaker_id':speaker_id,'voice_id':voice_id,'start':line['start'],'end':line['end'],'available_end':line['available_end'],'available_duration':line['available_duration'],'tts_end':line['start']+info.duration,'tts_duration':info.duration,'overlap_confirmed':line.get('overlap_confirmed',False),'overlap_speaker_ids':line.get('overlap_speaker_ids',[]),'path':str(path),'sample_rate':info.samplerate,'tts_speed':line['tts_speed'],'translated_text':line['translated_text'],'source_text':line['source_text']})
            print(f'Reused synthesized line {i+1}/{len(lines)}',flush=True); continue
        # Reset sampling to a stable speaker-specific seed for consistent style.
        if backend=='qwen3_mlx':
            mx.random.seed(1409+speaker_ids.index(speaker_id))
            results=list(model.generate(text=line['translated_text'],lang_code='Russian',ref_audio=ref['path'],ref_text=ref['text']))
            wav=np.asarray(results[0].audio).squeeze() if results else None
            sr=int(results[0].sample_rate) if results else 0
        elif backend=='qwen3':
            wavs,sr=model.generate_voice_clone(text=line['translated_text'],language='Russian',ref_audio=ref['path'],ref_text=ref['text'],x_vector_only_mode=True)
            wav=wavs[0]
        elif backend=='chatterbox':
            result=model.generate(text=line['translated_text'],language_id='ru',audio_prompt_path=ref['path'])
            wav=result.squeeze().cpu().numpy(); sr=model.sr
        if wav is None: raise RuntimeError(f'{backend} returned no audio for dialogue line {i}')
        sf.write(path,wav,sr)
        duration=sf.info(path).duration; available=float(line['available_duration'])
        if duration>.8: RATES.append(len(line['translated_text'])/duration)
        if duration>available+.03:
            ratio=duration/max(available,.1)
            if ratio>1.32:
                print('WARN',f"{line['segment_id']} ({speaker_id} → {voice_id}) is {duration:.2f}s but only {available:.2f}s are available; shortening is required before mixing.")
            paced=path.with_suffix('.paced.wav')
            tempo=min(ratio,float(cfg.get('tts_max_tempo',1.7))); af=f'atempo={tempo:.6f}'
            # Enforce maximum atempo of 1.7x as per requirements
            if tempo > 1.7:
                raise RuntimeError(f'Segment {line["segment_id"]} ({speaker_id} → {voice_id}) requires atempo > 1.7x to fit within {available:.2f}s. Cannot proceed.')
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(path),'-filter:a',af,'-ar',str(sr),str(paced)],check=True)
            paced.replace(path); duration=sf.info(path).duration
        resume[str(i)]={'text_sha256':digest,'sample_rate':int(sr),'speaker_id':speaker_id,'voice_id':voice_id,'tts_duration':duration}
        tmp=resume_path.with_suffix('.tmp'); tmp.write_text(json.dumps(resume,ensure_ascii=False,indent=2)); tmp.replace(resume_path)
        manifest.append({'i':i,'segment_id':line['segment_id'],'speaker_id':speaker_id,'voice_id':voice_id,'start':line['start'],'end':line['end'],'available_end':line['available_end'],'available_duration':line['available_duration'],'tts_end':line['start']+duration,'tts_duration':duration,'overlap_confirmed':line.get('overlap_confirmed',False),'overlap_speaker_ids':line.get('overlap_speaker_ids',[]),'path':str(path),'sample_rate':sr,'tts_speed':line['tts_speed'],'translated_text':line['translated_text'],'source_text':line['source_text']})
        print(f"segment {i+1:03d} speaker={speaker_id} voice={voice_id} start={line['start']:.2f} end={line['start']+duration:.2f} slot_end={line['available_end']:.2f}",flush=True)
    del model; gc.collect()
    if backend=='qwen3_mlx':
        try: mx.clear_cache()
        except Exception: pass
    if backend=='qwen3' and torch.backends.mps.is_available(): torch.mps.empty_cache()
    write_json(WORK/'tts_manifest.json',{'backend':backend,'voice_mode':'one persistent native Russian reference per speaker' if backend=='qwen3_mlx' else 'one persistent source reference per speaker','voice_mapping':speaker_to_voice,'lines':manifest,'references':refs})
    print(f'Synthesized {len(manifest)} TTS lines')
    if RATES:
        import statistics; r={'chars_per_second':round(statistics.median(RATES),2),'n':len(RATES)}; write_json(WORK/'rate.json',r); write_json(WORK.parent/'_tts_rate.json',r); print('TTS natural rate:',r,flush=True)

if __name__=='__main__': main()
