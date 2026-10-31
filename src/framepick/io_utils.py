"""图像 IO 与轻量预处理。"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

__all__ = ["IMAGE_EXTS", "load_image", "save_image", "resize_long_side", "dhash", "to_data_uri"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_image(path: str | Path, max_side: int | None = None) -> tuple[np.ndarray, float]:
    """读图为 RGB float32 [0,1]。

    返回 ``(rgb, scale)``，scale 为"原图 → 工作图"的缩放系数，
    用它可以把手算出来的裁剪框映射回原图坐标。
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    ow, oh = img.size
    scale = 1.0
    if max_side and max(ow, oh) > max_side:
        scale = max_side / float(max(ow, oh))
        img = img.resize((max(1, int(ow * scale)), max(1, int(oh * scale))), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr, scale


def resize_long_side(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    s = max_side / max(img.size)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)


def save_image(rgb: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def to_data_uri(rgb: np.ndarray, max_side: int = 480, quality: int = 85) -> str:
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = resize_long_side(img, max_side)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def dhash(rgb: np.ndarray, size: int = 8) -> int:
    """差值哈希，用于检测近乎重复的连拍（汉明距离 ≤ 6 视为重复）。"""
    g = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114)
    img = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
