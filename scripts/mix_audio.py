#!/usr/bin/env python3
"""Mix generated speech, selected library effects, and optional ambience."""
import json, subprocess
from pathlib import Path
import soundfile as sf
from common import ROOT, WORK, read_json, write_json
from integrity import validate_tts_timeline

def raise_fd_limit():
    # ffmpeg opens every input file up front, and a long video means 500+ TTS clips plus
    # SFX and ambience. A default macOS terminal caps a process at 256 open descriptors,
    # so the mix died with "Too many open files" (exit 232 = -EMFILE) around input #253.
    # Raising the soft limit to the hard one here also covers the ffmpeg child process.
    try:
        import resource
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = 10240
    if hard != resource.RLIM_INFINITY:
        wanted = min(wanted, hard)
    if soft < wanted:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
        except (OSError, ValueError):
            pass

def run(cmd): subprocess.run(cmd,check=True)

# A long video yields 500+ audio clips, so a single ffmpeg invocation opens hundreds of
# descriptors at once and builds an enormous command line. On macOS a process launched from
# a terminal is capped at 256 open files (soft AND hard), so that monolithic call died with
# "Too many open files" (exit 232 = -EMFILE) no matter how high we raise the soft limit.
# Mixing in chunks keeps every ffmpeg process small, so the mix no longer depends on the fd
# limit at all; raise_fd_limit above stays only as an optimization. This is the cap used
# when the fd limit is high or unknown; safe_chunk_inputs() below shrinks it further for
# unusually low caps.
CHUNK_INPUTS = 64

def safe_chunk_inputs():
    # Pick the chunk size from the real fd cap instead of trusting a bare constant. Measured
    # on this project's ffmpeg: an N-input amix holds ~0.7 descriptors per input (long clips
    # close early) plus a ~15-descriptor baseline for the process itself. So the largest safe
    # N for a soft cap S is (S - 15) / 0.7. Clamp to [8, CHUNK_INPUTS] so we never go below a
    # workable chunk nor above the conservative default when the cap is high or unknown.
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft != resource.RLIM_INFINITY and soft > 15:
            budget = int((soft - 15) / 0.7)
            return max(8, min(CHUNK_INPUTS, budget))
    except (ImportError, OSError, ValueError):
        pass
    return CHUNK_INPUTS

def main():
    raise_fd_limit()
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
    clips=[]
    for line in manifest:
        clips.append({'path':line['path'],'delay':max(0,int(float(line['start'])*1000)),'volume':1.0})
    for event in fx:
        clips.append({'path':event['path'],'delay':max(0,int(event['start']*1000)),'volume':event.get('volume',0.18)})
    if not clips and not amb.get('enabled'): raise SystemExit('No generated audio to mix')
    # Mix the clips in small chunks (each ffmpeg process opens only CHUNK_INPUTS files),
    # placing every chunk on the full-length silence timeline so the pieces align on merge.
    parts=[]
    total=len(clips)
    step=safe_chunk_inputs()
    for ci,base in enumerate(range(0,total,step)):
        group=clips[base:base+step]
        inputs=[]; filters=[]; labels=[]
        for i,clip in enumerate(group):
            inputs+=['-i',clip['path']]; label=f'g{i}'
            filters.append(f'[{i}:a]aresample=48000,volume={clip["volume"]},adelay={clip["delay"]}|{clip["delay"]}[{label}]'); labels.append(f'[{label}]')
        filters.append(''.join(labels)+f'amix=inputs={len(labels)}:duration=longest:normalize=0,apad,atrim=0:{duration}[out]')
        part=WORK/f'mix_part_{ci:03d}.wav'
        run(['ffmpeg','-y','-v','error',*inputs,'-filter_complex',';'.join(filters),'-map','[out]',
             '-ar','48000','-ac','2','-c:a','pcm_s16le',str(part)])
        parts.append(part)
    # Ambience is a single looping bed; keep it in its own pass so a huge clip count never
    # shares a process with it.
    if amb.get('enabled'):
        candidate=amb['candidate']; path=ROOT/candidate['path']
        part=WORK/'mix_part_amb.wav'
        filt=f'[0:a]aresample=48000,volume={amb["volume"]},atrim=0:{duration}[out]'
        run(['ffmpeg','-y','-v','error','-stream_loop','-1','-i',str(path),'-filter_complex',filt,'-map','[out]',
             '-ar','48000','-ac','2','-c:a','pcm_s16le',str(part)])
        parts.append(part)
    # Sum the chunks. amix divides by the input count, so scale each part back up first.
    inputs=[]; filters=[]; labels=[]
    for i,part in enumerate(parts):
        inputs+=['-i',str(part)]; label=f's{i}'
        filters.append(f'[{i}:a]volume={len(parts)}[{label}]'); labels.append(f'[{label}]')
    filters.append(''.join(labels)+f'amix=inputs={len(labels)}:duration=longest:normalize=0,alimiter=limit=0.95[out]')
    out=WORK/'mix.wav'
    run(['ffmpeg','-y','-v','error',*inputs,'-filter_complex',';'.join(filters),'-map','[out]',
         '-ar','48000','-ac','2','-c:a','pcm_s16le',str(out)])
    for part in parts: part.unlink(missing_ok=True)
    print(f'Mixed audio: {out} ({total} clips in {len(parts)} chunks)')
if __name__=='__main__': main()
