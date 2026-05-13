#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PREVIEW_DIR = Path('/Users/huez/Documents/ttds/lecture-handouts/d-previews')
OUT = Path('/Users/huez/Documents/ttds/lecture-handouts/d-previews/contact-sheet.png')


def font(size: int):
    for path in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/System/Library/Fonts/Helvetica.ttc']:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def main() -> int:
    files = sorted(PREVIEW_DIR.glob('*-tutor-handout-1.png'))
    if not files:
        print('no preview files')
        return 1
    thumb_w = 360
    caption_h = 34
    margin = 18
    cols = 4
    thumbs = []
    for file in files:
        img = Image.open(file).convert('RGB')
        scale = thumb_w / img.width
        thumb_h = int(img.height * scale)
        img = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        thumbs.append((file, img))
    thumb_h = max(img.height for _, img in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new('RGB', (cols * thumb_w + (cols + 1) * margin, rows * (thumb_h + caption_h) + (rows + 1) * margin), 'white')
    draw = ImageDraw.Draw(sheet)
    fnt = font(18)
    for i, (file, img) in enumerate(thumbs):
        row, col = divmod(i, cols)
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + caption_h + margin)
        sheet.paste(img, (x, y))
        draw.text((x, y + thumb_h + 6), file.name.replace('-tutor-handout-1.png', ''), fill=(40,40,40), font=fnt)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print(OUT)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
