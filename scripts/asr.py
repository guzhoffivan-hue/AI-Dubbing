#!/usr/bin/env python3
"""Low-RAM ASR using Qwen3-ASR with 120s streaming chunks."""
import argparse, gc, math, os, torch
import soundfile as sf
from common import ROOT, WORK, config, write_json, load_env, chunk_text_records

CHUNK_SECONDS = 120.0  # 2 минуты — гарантированно влезает в 16GB без свопа

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audio', default=str(WORK/'audio.wav'))
    p.add_argument('--output', default=str(WORK/'asr.json'))
    a = p.parse_args()
    load_env(); cfg = config(); model_id = cfg.get('asr_model', 'Qwen3-ASR-1.7B')
    
    # Читаем метаданные без загрузки всего файла
    with sf.SoundFile(a.audio) as f:
        total_frames = len(f)
        samplerate = f.samplerate
        
    chunk_frames = int(CHUNK_SECONDS * samplerate)
    chunks_count = max(1, math.ceil(total_frames / chunk_frames))
    chunk_dir = WORK / 'asr_chunks'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Splitting into {chunks_count} safe chunks (120s each)...", flush=True)
    chunk_paths = []
    with sf.SoundFile(a.audio) as f:
        for idx in range(chunks_count):
            cpath = chunk_dir / f'chunk_{idx:04d}.wav'
            frames = f.read(chunk_frames)
            sf.write(cpath, frames, samplerate)
            chunk_paths.append(cpath)

    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(
        model_id, 
        dtype=torch.float16, 
        device_map='mps', 
        max_inference_batch_size=1, 
        max_new_tokens=2048
    )
    language = cfg.get('input_language', 'auto')
    language = None if language.lower() == 'auto' else language
    
    recognized_parts = []
    chunk_texts = []
    detected_lang = 'zh'
    
    for idx, cpath in enumerate(chunk_paths):
        result = model.transcribe(audio=str(cpath), language=language)
        item = result[0]
        text_chunk = item.text.strip()
        detected_lang = item.language or detected_lang
        if text_chunk:
            recognized_parts.append(text_chunk)
        # One entry per chunk, in chunk order, empty strings included: alignment needs the
        # text that belongs to each audio chunk, not a proportional share of the transcript.
        chunk_texts.append(text_chunk)
        print(f"ASR Chunk {idx+1}/{chunks_count} completed ({len(text_chunk)} chars)", flush=True)
        
        # Жесткая очистка кэша после каждого чанка
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    full_text = ''.join(recognized_parts)
    records = chunk_text_records(chunk_texts, full_text)
    write_json(a.output, {'model': model_id, 'audio': a.audio, 'language': detected_lang, 'text': full_text, 'segments': [], 'chunks': records})
    print(f"ASR Complete: {len(full_text)} characters in {len(records)} chunks saved to {a.output}", flush=True)

if __name__ == '__main__':
    main()
