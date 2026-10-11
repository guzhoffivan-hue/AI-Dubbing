import json
from pathlib import Path

work = Path('work/42503442096-1-192')

# 1. Фиксируем 1800s в audio_meta.json
meta_path = work / 'audio_meta.json'
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
meta['duration'] = 1800.0
meta_path.write_text(json.dumps(meta, indent=2))
print("[+] audio_meta.json установлен на 1800.0s")

# 2. Обновляем status.json
status_path = work / 'status.json'
st = json.loads(status_path.read_text())
st['extract_audio'] = 'done'
st['asr'] = 'done'
st['diarization'] = 'done'
st['alignment'] = 'done'
if '_signature' in st:
    st['_signature']['duration'] = 1800.0

pending_steps = ['segment_merge', 'translation', 'dialogue', 'tts', 'fit_tts', 'ambience', 'sfx', 'mix', 'mux']
for step in pending_steps:
    if step in st:
        st[step] = 'pending'

status_path.write_text(json.dumps(st, indent=2, ensure_ascii=False))
print("[+] status.json успешно обновлен для 1800s")
