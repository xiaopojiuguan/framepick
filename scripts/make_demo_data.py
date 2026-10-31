"""生成一套**合成**的"同场景连拍"演示数据，方便没有数据集时把流水线跑通。

每组合成图共用同一背景（天空/山脊/草地/太阳），只在三处做扰动，正好对应真实连拍里
最影响选片的三件事：

* 构图偏移 —— 主体在画面里左右/上下移动；
* 曝光抖动 —— 整体亮度、对比度变化；
* 对焦/抖动 —— 高斯模糊强度不同。

另外会写一份 ``ground_truth.json``，标明"应该选哪张"，方便跑指标。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1500, 1000


def _gradient(w: int, h: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for i, (c0, c1) in enumerate(zip(top, bottom)):
        arr[:, :, i] = np.linspace(c0, c1, h, dtype=np.float32)[:, None]
    return Image.fromarray(arr.astype(np.uint8))


def _scene(seed: int) -> Image.Image:
    rng = random.Random(seed)
    horizon = int(H * 0.62)

    sky = _gradient(W, H, (86, 150, 214), (206, 228, 240))
    img = Image.new("RGB", (W, H))
    img.paste(sky, (0, 0))

    d = ImageDraw.Draw(img, "RGBA")

    # 太阳 + 光晕
    sx, sy = int(W * rng.uniform(0.16, 0.3)), int(H * rng.uniform(0.16, 0.26))
    for r, alpha in ((190, 26), (130, 46), (86, 80)):
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(255, 238, 190, alpha))
    d.ellipse([sx - 46, sy - 46, sx + 46, sy + 46], fill=(255, 249, 226, 235))

    # 远山
    for k, (amp, base, col) in enumerate(((0.09, 0.06, (126, 152, 176)), (0.07, 0.13, (92, 122, 148)))):
        pts = [(0, horizon)]
        for x in range(0, W + 40, 40):
            y = horizon - int(H * base) - int(H * amp * math.sin(x / 300.0 + k * 1.7))
            pts.append((x, y))
        pts += [(W, horizon), (0, horizon)]
        d.polygon(pts, fill=col)

    # 草地
    ground = _gradient(W, H - horizon, (118, 152, 86), (72, 104, 58))
    img.paste(ground, (0, horizon))
    d = ImageDraw.Draw(img, "RGBA")

    # 一条小路（引导线）
    d.polygon(
        [(int(W * 0.44), H), (int(W * 0.52), horizon), (int(W * 0.58), horizon), (int(W * 0.72), H)],
        fill=(196, 178, 148, 220),
    )
    return img


def _person(img: Image.Image, cx: int, cy: int, scale: float) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    hh = int(240 * scale)
    body_w = int(84 * scale)
    # 身体
    d.polygon(
        [(cx - body_w // 2, cy), (cx + body_w // 2, cy), (cx + body_w // 3, cy - int(hh * 0.62)),
         (cx - body_w // 3, cy - int(hh * 0.62))],
        fill=(46, 52, 68, 255),
    )
    # 腿
    d.rectangle([cx - int(body_w * 0.34), cy, cx - int(body_w * 0.06), cy + int(hh * 0.3)], fill=(38, 44, 58, 255))
    d.rectangle([cx + int(body_w * 0.06), cy, cx + int(body_w * 0.34), cy + int(hh * 0.3)], fill=(38, 44, 58, 255))
    # 头
    r = int(30 * scale)
    d.ellipse([cx - r, cy - int(hh * 0.62) - 2 * r, cx + r, cy - int(hh * 0.62)], fill=(214, 176, 148, 255))
    # 影子
    d.ellipse([cx - int(60 * scale), cy + int(hh * 0.3), cx + int(60 * scale), cy + int(hh * 0.36)], fill=(40, 50, 40, 90))


def make_series(out_dir: Path, n: int, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _scene(seed)
    rng = random.Random(seed * 977 + 13)
    meta = {"series": out_dir.name, "photos": [], "best": None}

    # 定义"一张理想图"的参数，其余围绕它抖动
    ideal = {"cx": int(W * 0.66), "cy": int(H * 0.74), "blur": 0.0, "bright": 1.0, "contrast": 1.0}
    scores = []
    for i in range(n):
        if i == 0:
            p = dict(ideal)
            bad = 0.0
        else:
            p = {
                "cx": int(ideal["cx"] + rng.choice([-1, 1]) * rng.uniform(0.06, 0.20) * W),
                "cy": int(ideal["cy"] + rng.uniform(-0.04, 0.04) * H),
                "blur": rng.choice([0.7, 1.5, 2.8]),
                "bright": rng.choice([0.76, 0.86, 1.14]),
                "contrast": rng.uniform(0.88, 1.0),
            }
            bad = abs(p["cx"] / W - 0.66) + p["blur"] * 0.25 + abs(p["bright"] - 1.0)
        img = base.copy()
        _person(img, p["cx"], p["cy"], 1.0)
        if p["blur"] > 0:
            img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip((arr - 0.5) * p["contrast"] + 0.5, 0, 1)
        arr = np.clip(arr * p["bright"], 0, 1)
        noise = np.random.default_rng(seed * 100 + i).normal(0, 0.012, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 1)
        name = f"IMG_{i + 1:03d}.jpg"
        Image.fromarray((arr * 255).astype(np.uint8)).save(out_dir / name, quality=92)
        meta["photos"].append(name)
        scores.append((bad, name))

    meta["best"] = min(scores)[1]
    (out_dir / "ground_truth.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--series", type=int, default=2)
    ap.add_argument("--photos", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.out)
    for s in range(args.series):
        meta = make_series(root / f"series_{s + 1:02d}", args.photos, seed=s + 1)
        print(f"[ok] {root / ('series_%02d' % (s + 1))}  ground truth = {meta['best']}")


if __name__ == "__main__":
    main()
