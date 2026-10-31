"""结果可视化：把组内排名画成一张对比拼图（答辩/报告直接可用）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io_utils import load_image

__all__ = ["contact_sheet", "draw_crop"]

FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def draw_crop(rgb: np.ndarray, box: tuple[int, int, int, int], color=(255, 64, 64), width: int = 4) -> Image.Image:
    img = to_pil(rgb).copy()
    d = ImageDraw.Draw(img)
    d.rectangle(list(box), outline=color, width=width)
    return img


def contact_sheet(result, out_path: str | Path, cell: int = 420) -> Path:
    """按 ranking 顺序画网格：首行标注 BEST，并画出取景框。"""
    items = result.ranking
    n = len(items)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    pad, header = 16, 46
    W = cols * (cell + pad) + pad
    H = rows * (cell + header + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    f_title = _font(20)
    f_small = _font(16)

    for i, pr in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + header + pad)
        img, _ = load_image(pr.path)  # 原图：裁剪框坐标要按 scale 映射回原图尺度
        thumb = to_pil(img)
        thumb.thumbnail((cell, cell), Image.LANCZOS)
        tw, th = thumb.size

        if i == 0:
            draw.rectangle([x - 2, y - 2, x + cell + 2, y + header + cell + 2], outline=(255, 200, 60), width=3)
            tag, color = "BEST", (255, 200, 60)
        else:
            tag, color = f"#{i + 1}", (200, 200, 200)

        draw.text((x + 6, y + 6), f"{tag}", font=f_title, fill=color)
        draw.text(
            (x + 76, y + 8),
            f"{pr.file}  util={pr.utility:.3f}",
            font=f_small,
            fill=(210, 210, 210),
        )
        canvas.paste(thumb, (x + (cell - tw) // 2, y + header + (cell - th) // 2))

        # 取景框（工作图坐标 → 原图坐标 → 缩略图坐标）
        sc = pr.scale or 1.0
        x1, y1, x2, y2 = (v / sc for v in pr.box)
        sx, sy = tw / img.shape[1], th / img.shape[0]
        box = (
            x + (cell - tw) // 2 + x1 * sx,
            y + header + (cell - th) // 2 + y1 * sy,
            x + (cell - tw) // 2 + x2 * sx,
            y + header + (cell - th) // 2 + y2 * sy,
        )
        draw.rectangle(box, outline=(255, 90, 90), width=2)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out
