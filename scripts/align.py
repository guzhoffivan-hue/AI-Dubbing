#!/usr/bin/env python3
"""Run Qwen3 ForcedAligner with bounded chunking aligned to ASR chunks."""
import argparse, gc, torch
from pathlib import Path
import soundfile as sf
from common import ROOT, WORK, config, read_json, write_json, load_env, chunk_offsets, resolve_chunk_texts

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audio', default=str(WORK/'audio.wav'))
    p.add_argument('--asr', default=str(WORK/'asr.json'))
    p.add_argument('--output', default=str(WORK/'alignment.json'))
    a = p.parse_args()
    load_env(); cfg = config(); source = read_json(a.asr); text = source['text']
    if not text: raise SystemExit('ASR transcript is empty')
    
    from qwen_asr import Qwen3ForcedAligner
    align_model_name = cfg.get('align_model', 'Qwen/Qwen3-ForcedAligner-0.6B')
    model = Qwen3ForcedAligner.from_pretrained(align_model_name, dtype=torch.float16, device_map='mps')
    
    chunk_dir = WORK / 'asr_chunks'
    chunk_files = sorted(chunk_dir.glob('chunk_*.wav')) if chunk_dir.exists() else []
    
    words = []
    
    # Если чанков нет (короткое видео) или файл < 300с — выравниваем напрямую
    if not chunk_files:
        items = model.align(audio=a.audio, text=text, language=source['language'])[0]
        words = [{'text': x.text, 'start': float(x.start_time), 'end': float(x.end_time)} for x in items]
    else:
        # Chunked alignment. Offsets are the cumulative real durations of the chunks
        # (see common.chunk_offsets), never an independent hard-coded chunk length:
        # a magic constant silently corrupts the whole timeline whenever asr.py
        # changes CHUNK_SECONDS.
        chunk_durations=[sf.info(str(p)).duration for p in chunk_files]
        offsets=chunk_offsets(chunk_durations)
        total_chunks = len(chunk_files)
        chunk_texts, used_recorded = resolve_chunk_texts(source, text, total_chunks)
        if used_recorded:
            print(f'Alignment text: per-chunk ASR transcripts for {total_chunks} chunks.', flush=True)
        else:
            print('WARNING: asr.json has no "chunks" field, so the transcript is split '
                  'proportionally by character count. Text boundaries then drift from the audio '
                  'boundaries by up to ~10 s and leave untimed tails before each chunk boundary, '
                  'so this alignment is approximate. Re-run asr.py to record per-chunk texts.', flush=True)
        
        for idx, cpath in enumerate(chunk_files):
            start_offset = offsets[idx]
            part_text = chunk_texts[idx]
            
            if part_text:
                try:
                    items = model.align(audio=str(cpath), text=part_text, language=source['language'])[0]
                    for x in items:
                        words.append({
                            'text': x.text,
                            'start': round(float(x.start_time) + start_offset, 3),
                            'end': round(float(x.end_time) + start_offset, 3)
                        })
                except Exception as e:
                    # В случае расхождения длительностей — безопасный линейный fallback таймингов
                    dur = chunk_durations[idx]
                    step = dur / max(len(part_text), 1)
                    for i_c, char in enumerate(part_text):
                        words.append({
                            'text': char,
                            'start': round(start_offset + i_c * step, 3),
                            'end': round(start_offset + (i_c + 1) * step, 3)
                        })
                
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            gc.collect()
            print(f'Aligned chunk {idx+1}/{total_chunks}', flush=True)

    del model; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    
    write_json(a.output, {'model': align_model_name, 'language': source['language'], 'text': text, 'words': words})
    print(f'Aligned {len(words)} tokens; saved {a.output}')

if __name__ == '__main__':
    main()
