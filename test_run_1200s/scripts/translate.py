#!/usr/bin/env python3
"""Context-aware, speaker-aware Chinese to Russian dialogue translation."""
import argparse, json, re, difflib
from common import ROOT, WORK, config, read_json, write_json, load_env, speech_rate

def clean_answer(text):
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
    text=re.sub(r'^(?:<<<\d+>>>|\d+[.)])\s*','',text).strip()
    return text.strip('"“” «»\n ')

# Characters of Russian accepted per second of source slot. Set in main() from the
# measured speech rate and the configured tempo target. The old hard-coded 13.5 forced
# atempo <= 1.04, so the model deleted facts to fit; tts.py can compress up to
# tts_max_tempo (1.7x) afterwards, which is what the budget should exploit.
CHAR_LIMIT_PER_SECOND = 13.5

def make_prompt(line, previous, following, char_limit, duration):
    max_chars = max(10, int(duration * CHAR_LIMIT_PER_SECOND))
    char_limit = min(char_limit, max_chars) if char_limit else max_chars
    payload={'current':{'speaker_id':line['speaker_id'],'acoustic_voice_type':line.get('voice_type','unknown'),'source_chinese':line['source_text']},'previous_lines_for_context_only':previous,'following_lines_for_context_only':following,'target_duration_seconds':round(duration,2),'max_russian_characters_including_spaces':char_limit,'glossary_names_use_exactly':GLOSSARY,'story_summary_and_characters':SUMMARY}
    system=('You are a professional Chinese-to-Russian audiovisual dialogue translator. Translate the CURRENT Chinese utterance into natural, idiomatic spoken Russian. Preserve exactly its meaning, facts, point of view, names, tone, and speaker intent. Use neighboring lines only to resolve references and maintain continuity; translate only the current line. Never copy a neighboring Russian line, combine different speakers, add events, invent dialogue, or fill in missing facts. Keep direct speech in the same person and tense as the source. Preserve every number, quantity and unit exactly as stated: a duration expressed in months must stay a duration, never a month name, and a number must never change value. Never omit a clause that carries a fact. Return only the Russian utterance, without analysis, labels, or quotation marks.')
    user=('Context follows. The acoustic voice type is only a cue for Russian grammatical agreement when the line itself identifies its speaker; do not infer story facts or character identity from voice type. Use simple, fluent Russian suitable for speech. CRITICAL: Your translation MUST fit within the exact length limit. Be concise.\n'+json.dumps(payload,ensure_ascii=False))
    return [{'role':'system','content':system},{'role':'user','content':user}]

GLOSSARY={}; SUMMARY=''
def build_summary(model,tokenizer,lines):
    global SUMMARY
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    text='\n'.join(l.get('source_text', '') or l.get('text', '') for l in lines)[:6000]
    msgs=[{'role':'system','content':'Read this transcript (source language) and write a compact briefing in Russian, max 120 words: who the speakers or characters are and how they relate, the setting, what is happening, slang and running jokes. Use facts from the text only; do not invent.'},{'role':'user','content':text}]
    try:
        prompt=tokenizer.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False,enable_thinking=False)
        SUMMARY=re.sub(r'<think>.*?</think>','',generate(model,tokenizer,prompt=prompt,max_tokens=300,verbose=False,sampler=make_sampler(temp=0.0)),flags=re.S).strip()
    except Exception as e: print('summary skipped:',type(e).__name__,flush=True)
    write_json(WORK/'summary.json',{'summary':SUMMARY}); print('summary:',SUMMARY[:200],flush=True)

def build_glossary(model,tokenizer,lines):
    global GLOSSARY
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    text='\n'.join(l.get('source_text', '') or l.get('text', '') for l in lines)[:6000]
    msgs=[{'role':'system','content':'Extract proper names (people, nicknames, places, brands) from the source text. Give each ONE fixed Russian rendering (standard practical transcription for the source language; established Russian form for known places). Output ONLY a JSON object that maps each name, written exactly as it appears in the source text, to its Cyrillic Russian form. Output {} if the text contains no proper names.'},{'role':'user','content':text}]
    try:
        prompt=tokenizer.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False,enable_thinking=False)
        out=re.sub(r'<think>.*?</think>','',generate(model,tokenizer,prompt=prompt,max_tokens=500,verbose=False,sampler=make_sampler(temp=0.0)),flags=re.S)
        m=re.search(r'\{.*\}',out,re.S); GLOSSARY={k:v for k,v in (json.loads(m.group(0)) if m else {}).items() if isinstance(v,str) and k in text}
    except Exception as e: print('glossary skipped:',type(e).__name__,flush=True)
    write_json(WORK/'glossary.json',GLOSSARY); print('glossary:',GLOSSARY,flush=True)

def apply_glossary_override():
    """Terms supplied by the user win over the generated glossary.

    A wrong name in the generated glossary is repeated in every line that mentions it,
    so an override file is the cheap, reviewable way to fix names and domain terms
    (the 1200 s test had three of four generated names wrong).
    """
    global GLOSSARY
    path=WORK/'glossary_override.json'
    if not path.exists():
        return
    try:
        override=json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f'glossary override ignored: {type(exc).__name__}: {exc}',flush=True); return
    if not isinstance(override,dict):
        print('glossary override ignored: not a JSON object',flush=True); return
    merged={key:value for key,value in override.items() if isinstance(key,str) and isinstance(value,str)}
    GLOSSARY={**GLOSSARY,**merged}
    write_json(WORK/'glossary.json',GLOSSARY)
    print(f'glossary override applied: {len(merged)} terms -> {merged}',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default=str(WORK/'segments.json')); a=p.parse_args()
    load_env(); cfg=config(); raw=read_json(a.input); lines=raw['lines']
    # Normalize incoming segment dicts to the keys expected downstream (source_text, segment_id, speaker_id)
    for idx, l in enumerate(lines):
        if 'source_text' not in l:
            l['source_text'] = l.get('text', '')
        if 'segment_id' not in l:
            l['segment_id'] = l.get('line_id', f'LINE_{idx+1:04d}')
        if 'speaker_id' not in l:
            l['speaker_id'] = l.get('speaker', 'SPEAKER_00')
    print(f'Normalized {len(lines)} lines; source_text present: {all("source_text" in l for l in lines)}', flush=True)
    if not lines: raise SystemExit('No speaker-labelled dialogue segments to translate')
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model_id=cfg.get('translation_model','mlx-community/Qwen3-8B-4bit')
    model,tokenizer=load(model_id); output=[]; raw_log=[]; RATE=speech_rate(); print(f'speech rate {RATE} chars/s',flush=True)
    global CHAR_LIMIT_PER_SECOND
    CHAR_LIMIT_PER_SECOND=RATE*float(cfg.get('tts_target_tempo',1.5))
    print(f'character budget: {CHAR_LIMIT_PER_SECOND:.1f} chars per second of slot '
          f'(speech rate {RATE} x tempo target {cfg.get("tts_target_tempo",1.5)})',flush=True)
    build_glossary(model,tokenizer,lines); apply_glossary_override(); build_summary(model,tokenizer,lines)
    for i,line in enumerate(lines):
        print(f"[{i+1}/{len(lines)}] Translating {line['segment_id']}...", flush=True)
        start=float(line['start']); end=float(line['end']);
        if i+1<len(lines) and not line.get('overlap_confirmed'):
            slot=max(.25,float(lines[i+1]['start'])-start)
        else: slot=max(.25,end-start)
        # Leave enough room for natural Russian, then let the TTS pacing guard
        # stretch the final audio modestly without crossing the next start.
        max_chars=max(8,int(slot*RATE*float(cfg.get('tts_target_tempo',1.35))))
        prev=[{'speaker_id':x['speaker_id'],'voice_type':x.get('voice_type','unknown'),'source_chinese':x['source_text'],'russian_translation':x.get('translated_text','')} for x in output[max(0,i-3):i]]
        following=[{'speaker_id':x['speaker_id'],'voice_type':x.get('voice_type','unknown'),'source_chinese':x['source_text']} for x in lines[i+1:min(len(lines),i+4)]]
        messages=make_prompt(line,prev,following,max_chars,slot)
        try: prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError: prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+'\n/no_think\n'
        try:
            answer=clean_answer(generate(model,tokenizer,prompt=prompt,max_tokens=384,verbose=False,sampler=make_sampler(temp=0.0)))
        except Exception as e:
            print(f'ERROR generation failed for {line["segment_id"]}: {e}', flush=True)
            answer=''
        attempts=1; first=answer
        while (len(answer)>max_chars or (cfg.get('output_language')=='ru' and re.search('[A-Za-z]',answer))) and attempts<2:
            short_messages=[{'role':'system','content':'You are editing a Russian dubbing line to fit a time slot. Make the wording tighter and more compact: use active voice, drop filler and repetition. You MUST keep EVERY fact, name and number of the draft and never delete a clause. Count characters before answering and output only the Russian line.'},{'role':'user','content':f'MAXIMUM {int(max_chars)} characters including spaces; the answer must not be longer. Do not drop facts to reach it.\nCURRENT CHINESE: {line["source_text"]}\nSTORY: {SUMMARY}\nRUSSIAN DRAFT: {answer}'}]
            try: short_prompt=tokenizer.apply_chat_template(short_messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            except TypeError: short_prompt=tokenizer.apply_chat_template(short_messages,tokenize=False,add_generation_prompt=True)+'\n/no_think\n'
            try:
                answer=clean_answer(generate(model,tokenizer,prompt=short_prompt,max_tokens=192,verbose=False,sampler=make_sampler(temp=0.0)))
            except Exception as e:
                print(f'ERROR retry generation failed for {line["segment_id"]}: {e}', flush=True)
                answer=''
            attempts+=1
        if any(difflib.SequenceMatcher(None,answer,x['russian_translation']).ratio()>0.7 for x in prev if x.get('russian_translation')): answer=first
        if not answer: raise SystemExit(f'Translation returned an empty line for {line["segment_id"]}')
        if len(answer)>max_chars:
            print(f'WARN {line["segment_id"]}: {len(answer)}>{max_chars} chars, atempo will fit it',flush=True)
        item={**line,'target_duration_seconds':round(slot,3),'max_russian_characters':max_chars,'translated_text':answer,'translation_attempts':attempts}
        output.append(item); raw_log.append({'segment_id':line['segment_id'],'speaker_id':line['speaker_id'],'source_text':line['source_text'],'context':{'previous':prev,'following':following},'target_duration':round(slot,3),'translation':answer,'attempts':attempts})
        print(f"Translated {i+1}/{len(lines)} {line['segment_id']} speaker={line['speaker_id']} {len(answer)}/{max_chars} chars",flush=True)
        # Intermediate save every 50 lines to allow resume
        if (i+1) % 50 == 0:
            write_json(WORK/'translated_segments.json',{'model':model_id,'lines':output})
            print(f'Intermediate save at line {i+1}', flush=True)
    del model,tokenizer
    write_json(WORK/'translated_segments.json',{'model':model_id,'lines':output})
    write_json(WORK/'translation.json',{'model':model_id,'lines':output})
    write_json(WORK/'translation_debug.json',{'model':model_id,'segments':raw_log})
    print(f'Translated {len(output)} speaker-labelled segments using neighboring context; saved translated_segments.json',flush=True)
if __name__=='__main__': main()
