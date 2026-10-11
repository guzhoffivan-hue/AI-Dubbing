#!/usr/bin/env python3
"""Apply per-segment speech budgets before TTS, preserving source start times."""
from common import WORK, read_json, write_json, speech_rate
from integrity import validate_translation

def main():
    # A partial or leftover translation must never become dialogue budgets for TTS.
    validate_translation(WORK)
    data=read_json(WORK/'translated_segments.json'); lines=sorted(data['lines'],key=lambda x:(float(x['start']),float(x['end']))); out=[]
    for i,line in enumerate(lines):
        start=float(line['start']); end=float(line['end']); allow=bool(line.get('overlap_confirmed'))
        if i+1<len(lines) and not allow:
            # Reserve the interval up to the next non-overlapping source line.
            available_end=max(start+.15,float(lines[i+1]['start']))
        else: available_end=max(start+.15,end)
        available=max(.15,available_end-start)
        text=line['translated_text'].strip(); estimated=len(text)/speech_rate()
        speed=round(max(1.0,estimated/available),2)
        out.append({**line,'start':start,'end':end,'speaker_id':line.get('speaker_id',line.get('speaker','UNKNOWN')),'voice_id':None,'available_end':round(available_end,3),'available_duration':round(available,3),'tts_speed':speed,'overlap_confirmed':allow})
    result={'lines':out,'timing_policy':'source start preserved; available duration ends at next line start unless diarization confirms simultaneous speakers'}
    write_json(WORK/'dialogue.json',result); write_json(WORK/'dialogue_segments.json',result)
    print(f'Prepared {len(out)} dialogue segments with non-overlap speech budgets',flush=True)
    for x in out: print(f"{x['segment_id']} {x['start']:.2f}-{x['available_end']:.2f} speaker={x['speaker_id']} overlap={x['overlap_confirmed']} budget={x['available_duration']:.2f}s",flush=True)
if __name__=='__main__': main()
