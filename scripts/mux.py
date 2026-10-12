#!/usr/bin/env python3
"""Mux video frames with newly generated Russian audio and optional Russian subtitles."""
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from common import ROOT, WORK, INPUT_VIDEO, OUTPUT_DIR, config, read_json, write_json


def _format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def _collect_subtitle_cues() -> list[dict]:
    cues = []
    mix_report = WORK / 'audio_mix_report.json'
    tts_manifest = WORK / 'tts_manifest.json'
    dialogue_file = WORK / 'dialogue.json'

    raw_items = []
    if mix_report.exists():
        data = read_json(mix_report)
        raw_items = data.get('items', [])
    elif tts_manifest.exists():
        data = read_json(tts_manifest)
        raw_items = data.get('lines', [])
    elif dialogue_file.exists():
        data = read_json(dialogue_file)
        raw_items = data.get('lines', [])

    for item in raw_items:
        if item.get('skipped_echo_duplicate') or item.get('skipped_stutter_prefix'):
            continue
        text = str(item.get('translated_text') or item.get('text') or '').strip()
        if not text:
            continue
        start = float(item.get('final_start', item.get('start', 0.0)))
        end = float(item.get('final_end', item.get('tts_end', item.get('end', start + 1.5))))
        if end <= start + 0.08:
            end = start + 0.8
        if cues and start < cues[-1]['end']:
            cues[-1]['end'] = max(cues[-1]['start'] + 0.15, start - 0.02)
        cues.append({'start': round(start, 3), 'end': round(end, 3), 'text': text})
    return cues


def _write_srt(cues: list[dict], srt_path: Path) -> None:
    blocks = []
    for idx, cue in enumerate(cues, 1):
        t0 = _format_srt_time(cue['start'])
        t1 = _format_srt_time(cue['end'])
        blocks.append(f"{idx}\n{t0} --> {t1}\n{cue['text']}\n")
    srt_path.write_text('\n'.join(blocks), encoding='utf-8')


def _probe_video_dimensions(video_path: Path) -> tuple[int, int]:
    try:
        out = subprocess.check_output(
            [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'json', str(video_path),
            ],
            text=True,
        )
        streams = json.loads(out).get('streams', [])
        if streams:
            return int(streams[0].get('width', 1280)), int(streams[0].get('height', 720))
    except Exception:
        pass
    return 1280, 720


def _load_subtitle_font(font_size: int):
    from PIL import ImageFont
    candidates = [
        '/System/Library/Fonts/HelveticaNeue.ttc',
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
        '/System/Library/Fonts/SFNS.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_subtitle_text(draw, text: str, font, max_px: int) -> str:
    words = text.split()
    if not words:
        return text
    lines = []
    cur = words[0]
    for w in words[1:]:
        cand = f'{cur} {w}'
        bbox = draw.textbbox((0, 0), cand, font=font)
        if (bbox[2] - bbox[0]) <= max_px:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    if len(lines) > 2:
        # Re-balance into 2 lines if possible
        mid = len(text) // 2
        wrapped = textwrap.wrap(text, width=max(24, int(len(text) * 0.56)))
        return '\n'.join(wrapped[:3])
    return '\n'.join(lines)


def _mux_with_burned_subtitles(src: Path, audio: Path, srt_path: Path, out: Path, cues: list[dict]) -> bool:
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError:
        return False

    vw, vh = _probe_video_dimensions(src)
    font_size = max(22, int(min(vw, vh) * 0.042))
    strip_h = max(140, int(font_size * 4.6))
    bottom_margin = max(16, int(vh * 0.032))
    font = _load_subtitle_font(font_size)

    audio_dur = 0.0
    if (WORK / 'audio_meta.json').exists():
        audio_dur = float(read_json(WORK / 'audio_meta.json').get('duration', 0.0))
    if audio_dur <= 0.0 and cues:
        audio_dur = cues[-1]['end'] + 2.0

    with tempfile.TemporaryDirectory(prefix='aidub_subs_') as tmp_dir:
        tmp = Path(tmp_dir)
        blank_png = tmp / 'blank.png'
        Image.new('RGBA', (vw, strip_h), (0, 0, 0, 0)).save(blank_png)

        concat_lines = []
        cursor = 0.0
        for idx, cue in enumerate(cues):
            start = max(cursor, float(cue['start']))
            end = float(cue['end'])
            if end <= start + 0.04:
                continue
            gap = start - cursor
            if gap > 0.01:
                concat_lines.append(f"file '{blank_png}'\nduration {gap:.4f}")

            strip = Image.new('RGBA', (vw, strip_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(strip)
            wrapped = _wrap_subtitle_text(draw, cue['text'], font, int(vw * 0.84))
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align='center', spacing=6)
            ink_w = int(round(bbox[2] - bbox[0]))
            ink_h = int(round(bbox[3] - bbox[1]))

            pad_x = max(22, int(font_size * 0.75))
            pad_y = max(12, int(font_size * 0.42))
            pill_w = int(ink_w + 2 * pad_x)
            pill_h = int(ink_h + 2 * pad_y)
            radius = pill_h // 2

            x0 = int((vw - pill_w) // 2)
            y0 = int(max(10, strip_h - pill_h - 14))
            x1 = x0 + pill_w
            y1 = y0 + pill_h

            # 1. Floating drop shadow underneath the capsule
            shadow = Image.new('RGBA', (vw, strip_h), (0, 0, 0, 0))
            sdraw = ImageDraw.Draw(shadow)
            sdraw.rounded_rectangle([x0, y0 + 4, x1, y1 + 4], radius=radius, fill=(0, 0, 0, 115))
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=6))
            strip = Image.alpha_composite(strip, shadow)

            # 2. Convex 3D Capsule (like .btn-dub: glossy top crown, top-only white inset rim, bottom-only dark inset rim)
            pill_mask = Image.new('L', (pill_w, pill_h), 0)
            mdraw = ImageDraw.Draw(pill_mask)
            mdraw.rounded_rectangle([0, 0, pill_w - 1, pill_h - 1], radius=radius, fill=255)
            mask_f = np.asarray(pill_mask, dtype=np.float32) / 255.0

            t = np.linspace(0.0, 1.0, pill_h, dtype=np.float32)[:, None]
            # Base convex gradient: bright slate-grey crown -> rich charcoal body -> deep bottom edge
            r = np.interp(t, [0.0, 0.45, 0.55, 1.0], [96.0, 52.0, 36.0, 20.0])
            g = np.interp(t, [0.0, 0.45, 0.55, 1.0], [102.0, 56.0, 39.0, 22.0])
            b = np.interp(t, [0.0, 0.45, 0.55, 1.0], [114.0, 66.0, 48.0, 28.0])

            # True CSS-like `inset 0 2px 1px rgba(255,255,255,0.85)` (top crescent only!)
            inv_mask = Image.fromarray(((1.0 - mask_f) * 255.0).astype(np.uint8), 'L')
            top_hilite_img = Image.new('L', (pill_w, pill_h), 255)
            top_hilite_img.paste(inv_mask, (0, 2))
            top_hilite_img = top_hilite_img.filter(ImageFilter.GaussianBlur(radius=0.8))
            top_rim = (np.asarray(top_hilite_img, dtype=np.float32) / 255.0) * mask_f * np.clip(1.0 - t * 1.6, 0.0, 1.0)

            # True CSS-like `inset 0 -2px 1.5px rgba(0,0,0,0.55)` (bottom crescent only!)
            bot_shade_img = Image.new('L', (pill_w, pill_h), 255)
            bot_shade_img.paste(inv_mask, (0, -2))
            bot_shade_img = bot_shade_img.filter(ImageFilter.GaussianBlur(radius=1.0))
            bot_rim = (np.asarray(bot_shade_img, dtype=np.float32) / 255.0) * mask_f * np.clip((t - 0.4) * 1.7, 0.0, 1.0)

            # Blend top white specular highlight & bottom dark inner bevel
            hilite_strength = np.clip(top_rim * 0.82, 0.0, 1.0)
            shade_strength = np.clip(bot_rim * 0.55, 0.0, 1.0)
            r = r * (1.0 - shade_strength) + (255.0 - r) * hilite_strength
            g = g * (1.0 - shade_strength) + (255.0 - g) * hilite_strength
            b = b * (1.0 - shade_strength) + (255.0 - b) * hilite_strength

            a = np.interp(t, [0.0, 0.5, 1.0], [230.0, 224.0, 236.0])
            grad_arr = np.zeros((pill_h, pill_w, 4), dtype=np.uint8)
            grad_arr[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
            grad_arr[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)
            grad_arr[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)
            grad_arr[:, :, 3] = np.clip(a * mask_f, 0, 255).astype(np.uint8)
            pill_img = Image.fromarray(grad_arr, 'RGBA')

            # Subtle dark outer hairline border (like `border: 0.5px solid rgba(0,0,0,0.35)`)
            pdraw = ImageDraw.Draw(pill_img)
            pdraw.rounded_rectangle([0, 0, pill_w - 1, pill_h - 1], radius=radius, outline=(0, 0, 0, 95), width=1)

            strip.alpha_composite(pill_img, (x0, y0))
            draw = ImageDraw.Draw(strip)

            # 3. Exact geometric & optical centering (compensating for PIL's bbox[0] and bbox[1] offsets)
            tx = int(round(x0 + (pill_w - ink_w) / 2.0 - bbox[0]))
            ty = int(round(y0 + (pill_h - ink_h) / 2.0 - bbox[1] - 1))
            draw.multiline_text((tx, ty + 1), wrapped, font=font, fill=(0, 0, 0, 160), align='center', spacing=6)
            draw.multiline_text((tx, ty), wrapped, font=font, fill=(255, 255, 255, 255), align='center', spacing=6)

            sub_png = tmp / f'sub_{idx:04d}.png'
            strip.save(sub_png)
            dur = end - start
            concat_lines.append(f"file '{sub_png}'\nduration {dur:.4f}")
            cursor = end

        tail = max(1.0, audio_dur - cursor + 1.0)
        concat_lines.append(f"file '{blank_png}'\nduration {tail:.4f}")
        concat_lines.append(f"file '{blank_png}'")
        concat_file = tmp / 'sub_concat.txt'
        concat_file.write_text('\n'.join(concat_lines) + '\n', encoding='utf-8')

        start_offset = float(os.environ.get('AI_DUB_START', 0) or 0)
        cmd = ['ffmpeg', '-y', '-v', 'error']
        if start_offset > 0:
            cmd += ['-ss', str(start_offset)]
        cmd += [
            '-i', str(src),
            '-i', str(audio),
            '-f', 'concat', '-safe', '0', '-i', str(concat_file),
            '-i', str(srt_path),
            '-filter_complex', f'[0:v][2:v]overlay=0:H-h-{bottom_margin}:eof_action=pass[v]',
            '-map', '[v]',
            '-map', '1:a:0',
            '-map', '3:s:0',
            '-c:v', 'h264_videotoolbox', '-q:v', '68',
            '-c:a', 'aac', '-b:a', '192k',
            '-c:s', 'mov_text',
            '-metadata:s:s:0', 'language=rus',
            '-metadata:s:s:0', 'title=Русский',
            '-shortest',
            '-movflags', '+faststart',
            str(out),
        ]
        res = subprocess.run(cmd)
        return res.returncode == 0


def main():
    cfg = config()
    if cfg.get('use_original_audio', True):
        raise SystemExit('Safety check: original audio must remain disabled')

    env_subs = os.environ.get('AI_DUB_ENABLE_SUBTITLES')
    if env_subs is not None:
        enable_subtitles = env_subs.strip().lower() in ('1', 'true', 'yes', 'on')
    else:
        enable_subtitles = bool(cfg.get('enable_subtitles', False))

    src = INPUT_VIDEO or next((ROOT / 'input').glob('*.mp4'))
    audio = WORK / 'mix.wav'
    out = Path(os.environ.get('AI_DUB_OUTPUT_FILE', OUTPUT_DIR / 'test_dub.mp4'))
    out.parent.mkdir(parents=True, exist_ok=True)

    cues = _collect_subtitle_cues()
    srt_path = WORK / 'subtitles.srt'
    if cues:
        _write_srt(cues, srt_path)

    out_srt = out.with_suffix('.srt')
    if enable_subtitles and cues:
        shutil.copyfile(srt_path, out_srt)
        print(f'Saved SRT subtitles: {out_srt}', flush=True)
        if _mux_with_burned_subtitles(src, audio, srt_path, out, cues):
            write_json(WORK / 'mux_options.json', {'enable_subtitles': True})
            print(f'Created {out} (with Russian subtitles)')
            return
        # Fallback to fast stream copy + soft subtitles if VideoToolbox overlay wasn't used
        start_offset = float(os.environ.get('AI_DUB_START', 0) or 0)
        cmd = ['ffmpeg', '-y', '-v', 'error']
        if start_offset > 0:
            cmd += ['-ss', str(start_offset)]
        cmd += [
            '-i', str(src), '-i', str(audio), '-i', str(srt_path),
            '-map', '0:v:0', '-map', '1:a:0', '-map', '2:s:0',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
            '-c:s', 'mov_text', '-disposition:s:0', 'default+forced',
            '-metadata:s:s:0', 'language=rus',
            '-shortest', '-movflags', '+faststart', str(out),
        ]
        subprocess.run(cmd, check=True)
        write_json(WORK / 'mux_options.json', {'enable_subtitles': True})
        print(f'Created {out} (with soft subtitles)')
        return

    if out_srt.exists():
        try:
            out_srt.unlink()
        except OSError:
            pass
    start_offset = float(os.environ.get('AI_DUB_START', 0) or 0)
    cmd = ['ffmpeg', '-y', '-v', 'error']
    if start_offset > 0:
        cmd += ['-ss', str(start_offset)]
    cmd += [
        '-i', str(src), '-i', str(audio),
        '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
        '-shortest', '-movflags', '+faststart', str(out),
    ]
    subprocess.run(cmd, check=True)
    write_json(WORK / 'mux_options.json', {'enable_subtitles': False})
    print(f'Created {out}')


if __name__ == '__main__':
    main()
