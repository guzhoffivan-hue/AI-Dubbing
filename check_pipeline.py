import sys, os, json
from pathlib import Path

work = Path('work/42503442096-1-192')
scripts = Path('scripts')

print("="*60)
print("1. ВХОДНЫЕ ДАННЫЕ И ТАЙМИНГИ")
print("="*60)
for name in ['audio.wav', 'asr.json', 'alignment.json', 'diarization.json']:
    p = work / name
    if not p.exists():
        print(f"[-] {name}: ОТСУТСТВУЕТ")
        continue
    if name.endswith('.wav'):
        sz = p.stat().st_size / (1024 * 1024)
        print(f"[+] {name}: размер {sz:.1f} MB")
    elif name.endswith('.json'):
        d = json.loads(p.read_text())
        if name == 'asr.json':
            print(f"[+] asr.json: символов={len(d.get('text', ''))}")
        elif name == 'alignment.json':
            w = d.get('words', [])
            print(f"[+] alignment.json: слов={len(w)}, охват до {w[-1].get('end', 0):.2f}s" if w else "[]")
        elif name == 'diarization.json':
            t = d.get('turns') or d.get('segments') or []
            print(f"[+] diarization.json: turns={len(t)}, охват до {t[-1].get('end', 0):.2f}s" if t else "[]")

print("\n" + "="*60)
print("2. ТЕСТ MERGE_SEGMENTS (СКОЛЬКО РЕПЛИК БУДЕТ)")
print("="*60)
try:
    sys.path.append('scripts')
    import merge_segments
    asr = json.loads((work / 'asr.json').read_text())
    align = json.loads((work / 'alignment.json').read_text())
    diar = json.loads((work / 'diarization.json').read_text())
    words = align.get('words', [])
    words = merge_segments.attach_punctuation(asr.get('text', ''), words)
    diar_list = diar.get('turns') or diar.get('segments') or []
    words = merge_segments.assign_speakers(words, diar_list)
    lines = merge_segments.group_words(words)
    print(f"[+] Сгенерировано реплик: {len(lines)}")
    if lines:
        print(f"    Первая: {lines[0]['start']}s -> {lines[0]['end']}s ({lines[0]['speaker']})")
        print(f"    Последняя: {lines[-1]['start']}s -> {lines[-1]['end']}s ({lines[-1]['speaker']})")
except Exception as e:
    print(f"[-] Ошибка в merge_segments: {e}")

print("\n" + "="*60)
print("3. ПРОВЕРКА СКРИПТОВ (TTS, FIT, MIX, SFX, MUX)")
print("="*60)
for p in sorted(scripts.glob('*.py')):
    txt = p.read_text()
    alerts = []
    for word in ['atempo', 'tts_speed', 'duration', 'pad', 'shortest', 'max_tokens', '192', '210']:
        if word in txt:
            matches = [l.strip() for l in txt.splitlines() if word in l and not l.strip().startswith('#')]
            if matches:
                alerts.append(f"{word}: {matches[0][:80]}")
    if alerts:
        print(f"\n--- {p.name} ---")
        for a in alerts[:4]:
            print(f"  * {a}")
