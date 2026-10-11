"""Shared project paths, configuration, and safe JSON helpers."""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('AI_DUB_WORKDIR', ROOT / 'work')).resolve()
INPUT_VIDEO = Path(os.environ.get('AI_DUB_INPUT_VIDEO', '')).resolve() if os.environ.get('AI_DUB_INPUT_VIDEO') else None
OUTPUT_DIR = Path(os.environ.get('AI_DUB_OUTPUT_DIR', ROOT / 'output')).resolve()

def load_env():
    path = ROOT / '.env'
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))

def config():
    import yaml
    return yaml.safe_load((ROOT / 'config.yaml').read_text())

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def speech_rate(default=14.0):
    import json
    for p in (WORK/'rate.json', WORK.parent/'_tts_rate.json'):
        try: return float(json.load(open(p))['chars_per_second'])
        except Exception: pass
    return default

def chunk_offsets(durations):
    """Start offset of each sequential audio chunk, from the real chunk durations.

    Offset of chunk i is the sum of the durations of chunks 0..i-1. Deriving the
    offsets from measured durations (instead of a hard-coded chunk length) keeps the
    timeline correct when asr.py changes its chunk size.
    """
    offsets=[]; total=0.0
    for duration in durations:
        offsets.append(total)
        total+=float(duration)
    return offsets

def validate_chunk_texts(records, full_text, expected_count):
    """Return the per-chunk texts from an asr.json 'chunks' field, or raise.

    Checks that the records are in order, that each declared length matches its text,
    that the count matches the audio chunks, and that the texts reconstruct the full
    transcript exactly. A mismatch here would silently shift text between chunks, which
    is the defect this field exists to prevent, so it is an error and never a warning.
    """
    if not isinstance(records, list) or not records:
        raise ValueError('asr.json "chunks" must be a non-empty list')
    if len(records) != expected_count:
        raise ValueError(f'asr.json lists {len(records)} text chunks but {expected_count} audio chunks exist')
    texts=[]
    for position, record in enumerate(records):
        if not isinstance(record, dict) or 'text' not in record:
            raise ValueError(f'chunk record {position} has no text')
        if record.get('index') != position:
            raise ValueError(f'chunk records are out of order at position {position} (index={record.get("index")!r})')
        text=record['text']
        if not isinstance(text, str):
            raise ValueError(f'chunk {position}: text must be a string')
        if record.get('chars') != len(text):
            raise ValueError(f'chunk {position}: chars={record.get("chars")!r} but text has {len(text)} characters')
        texts.append(text)
    if ''.join(texts) != full_text:
        raise ValueError('chunk texts do not reconstruct the full transcript exactly')
    return texts

def chunk_text_records(chunk_texts, full_text):
    """Build the asr.json 'chunks' field from the per-chunk transcripts, checked."""
    records=[{'index': index, 'text': text, 'chars': len(text)} for index, text in enumerate(chunk_texts)]
    validate_chunk_texts(records, full_text, len(records))
    return records

def split_text_proportionally(text, parts):
    """Legacy fallback: cut the transcript into equal character blocks.

    This is the behaviour that shifted text across chunk boundaries, so callers must
    warn when they use it. Kept only for backward compatibility with old asr.json files.
    """
    if parts <= 0:
        return []
    average=len(text)/parts
    out=[]; cursor=0
    for index in range(parts):
        take=int(round((index+1)*average))-cursor if index < parts-1 else len(text)-cursor
        piece=text[cursor:cursor+take]
        cursor+=len(piece)
        out.append(piece)
    return out

def resolve_chunk_texts(asr_document, full_text, chunk_count):
    """Per-chunk texts for alignment: the recorded chunk texts when available.

    Returns (texts, used_recorded_chunks). Falls back to the proportional split only
    when the 'chunks' field is absent, so old artifacts keep working; a present but
    inconsistent field raises instead of silently misaligning text.
    """
    records=asr_document.get('chunks')
    if records:
        return validate_chunk_texts(records, full_text, chunk_count), True
    return split_text_proportionally(full_text, chunk_count), False
