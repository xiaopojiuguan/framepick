"""用**真实照片**生成报告配图。

两张图：

* 图 A：单张真实照片上的显著性 → 候选取景框 → 最终裁剪（对应报告 §3.2）
* 图 B：以真实照片为底合成的连拍组，走完整流水线后的组内选片结果（对应报告 §5）

照片来源：picsum.photos（Lorem Picsum，图片来自 Unsplash，Unsplash License 可自由使用）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framepick.candidates import generate_candidates  # noqa: E402
from framepick.features import candidate_features, crop  # noqa: E402
from framepick.fusion import fuse  # noqa: E402
from framepick.io_utils import load_image  # noqa: E402
from framepick.pipeline import FramePick  # noqa: E402
from framepick.saliency import saliency_map  # noqa: E402
from framepick.visualize import contact_sheet  # noqa: E402

REAL = ROOT / "data" / "real"
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "outputs" / "real"

INK = (17, 24, 39)
MUTED = (107, 114, 128)
RED = (220, 38, 38)
BLUE = (37, 99, 235)


def font(size: int, bold: bool = False):
    cands = [
        ("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        ("C:/Windows/Fonts/simhei.ttf"),
        ("C:/Windows/Fonts/simsun.ttc"),
    ]
    for p in cands:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def ensure_photo(name: str) -> Path:
    """本地没有就从 Lorem Picsum 取（图片来自 Unsplash，Unsplash License）。"""
    p = REAL / f"{name}.jpg"
    if p.exists() and p.stat().st_size > 10000:
        return p
    REAL.mkdir(parents=True, exist_ok=True)
    import urllib.request

    url = f"https://picsum.photos/id/{name.lstrip('p')}/1600/1200"
    print(f"[dl ] {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(p, "wb") as f:
        f.write(r.read())
    return p


def heatmap(sal: np.ndarray, base: np.ndarray | None = None, alpha: float = 0.55) -> Image.Image:
    """显著性热图（蓝→青→黄→红），可叠加到原图上。"""
    s = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    r = np.clip(1.5 - np.abs(4 * s - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * s - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * s - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    if base is not None:
        rgb = (1 - alpha) * base + alpha * rgb
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def panel(im: Image.Image, w: int) -> Image.Image:
    if im.width == w:
        return im
    return im.resize((w, max(1, int(im.height * w / im.width))), Image.LANCZOS)


def _chip(d: ImageDraw.ImageDraw, xy, text: str, fnt, bg, fg=(255, 255, 255)) -> None:
    """带底色的小标签，避免压在图上看不清。"""
    x, y = xy
    b = d.textbbox((0, 0), text, font=fnt)
    w, h = b[2] - b[0], b[3] - b[1]
    d.rectangle([x, y, x + w + 8, y + h + 8], fill=bg)
    d.text((x + 4, y + 4 - b[1]), text, font=fnt, fill=fg)


# --------------------------------------------------------------- 图 A


def figure_candidates(name: str = "p1025", top_show: int = 6) -> Path:
    path = ensure_photo(name)
    rgb, _ = load_image(path, max_side=1200)
    sal = saliency_map(rgb)
    cands = generate_candidates(rgb, sal, max_candidates=32)
    feats = [candidate_features(rgb, sal, c.box) for c in cands]
    utils = fuse(feats, mode="absolute")
    order = list(np.argsort(-np.asarray(utils)))
    best = cands[order[0]]
    best_feat = feats[order[0]]
    best_u = float(utils[order[0]])
    full_u = float(utils[[i for i, c in enumerate(cands) if c.kind == "full"][0]])

    W = 520
    H = rgb.shape[0]
    im0 = panel(Image.fromarray((rgb * 255).astype(np.uint8)), W)

    # ① 候选框：Top-6 用蓝色细框，最优用红色粗框
    im1 = im0.copy()
    d1 = ImageDraw.Draw(im1)
    sy = im1.height / H
    sx = im1.width / rgb.shape[1]
    for rank, idx in enumerate(order[:top_show], 1):
        c = cands[idx]
        x1, y1, x2, y2 = c.box
        box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        if rank == 1:
            continue
        d1.rectangle(box, outline=BLUE, width=2)
    x1, y1, x2, y2 = best.box
    d1.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline=RED, width=4)
    _chip(d1, (x1 * sx + 8, y1 * sy + 8), "#1 最优", font(15, True), RED)

    # ② 显著性热图
    im2 = panel(heatmap(sal, rgb), W)

    # ③ 最优裁剪
    sub = crop(rgb, best.box)
    im3 = panel(Image.fromarray((sub * 255).astype(np.uint8)), W)

    gap, cap = 16, 46
    rowh = max(im1.height, im2.height, im3.height)
    canvas = Image.new("RGB", (W * 3 + gap * 2, rowh + cap), "white")
    for i, im in enumerate((im1, im2, im3)):
        canvas.paste(im, (i * (W + gap), (rowh - im.height) // 2))
    d = ImageDraw.Draw(canvas)
    titles = [
        "① 原图 + 候选取景框（蓝：其余 Top-4，红：最优）",
        "② 谱残差显著性图（主体响应）",
        f"③ 最优裁剪  U={best_u:.3f}（整图 U={full_u:.3f}）",
    ]
    for i, t in enumerate(titles):
        d.text((i * (W + gap), rowh + 12), t, font=font(17), fill=INK)

    out = FIG / "real_candidates.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92)
    print(f"[图A] {out.name} {canvas.size}  best={best.box} util={best_u:.3f} "
          f"(q={best_feat['quality']:.2f} c={best_feat['composition']:.2f} "
          f"sub={best_feat['subject']:.2f}) full={full_u:.3f}")
    return out


# --------------------------------------------------------------- 图 B


def _adjust(arr: np.ndarray, gain: float, contrast: float) -> np.ndarray:
    out = (arr - 0.5) * contrast + 0.5
    return np.clip(out * gain, 0, 1)


def _blur(arr: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return arr
    im = Image.fromarray((arr * 255).astype(np.uint8))
    from PIL import ImageFilter

    im = im.filter(ImageFilter.GaussianBlur(r))
    return np.asarray(im, dtype=np.float32) / 255.0


def make_burst(src: str, out_dir: Path, n: int = 8, seed: int = 7,
               ideal_shot: int = 2) -> tuple[list[Path], int]:
    """以真实照片为底合成一组连拍：窗口微移 + 曝光/对比度/模糊扰动。

    第 ``ideal_shot`` 张被设为"人工首选"——主体落在三分点、对焦最实、曝光最准；
    其余张各有明显缺陷（过曝/欠曝、对比发灰、失焦）。
    """
    rng = np.random.default_rng(seed)
    rgb, _ = load_image(ensure_photo(src), max_side=1400)
    H, W = rgb.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # 主体质心（用显著性估计），用于把"最好的一张"锚定到三分点
    sal = saliency_map(rgb)
    ys, xs = np.mgrid[0:H, 0:W]
    wgt = sal + 1e-6
    scx = float((wgt * xs).sum() / wgt.sum())
    scy = float((wgt * ys).sum() / wgt.sum())

    base_w, base_h = W * 0.94, H * 0.94
    for i in range(n):
        if i == ideal_shot:
            bw, bh = base_w * 0.86, base_h * 0.86
            x = float(np.clip(scx - (2 / 3) * bw, 0, W - bw))
            y = float(np.clip(scy - (1 / 3) * bh, 0, H - bh))
            gain, contrast, blur = 1.02, 1.04, 0.0
        else:
            bw = base_w * rng.uniform(0.80, 1.0)
            bh = base_h * rng.uniform(0.80, 1.0)
            x = rng.uniform(0, W - bw)
            y = rng.uniform(0, H - bh)
            # 其余张：曝光/对比明显偏离、一定程度的失焦
            gain = float(rng.choice([0.62, 0.70, 1.38, 1.46]))
            contrast = float(rng.uniform(0.78, 0.88))
            blur = float(rng.choice([2.0, 2.6, 3.2, 3.8]))
        box = (int(x), int(y), int(x + bw), int(y + bh))
        sub = crop(rgb, box)
        sub = _adjust(sub, gain, contrast)
        sub = _blur(sub, int(round(blur)))
        # 轻微噪点，模拟高感光
        sub = np.clip(sub + rng.normal(0, 0.006, sub.shape), 0, 1)
        p = out_dir / f"IMG_{i + 1:03d}.jpg"
        Image.fromarray((sub * 255).astype(np.uint8)).save(p, quality=90)
        paths.append(p)
    return paths, ideal_shot


def figure_group(src: str = "p1025", ideal: int = 2) -> Path:
    out_dir = OUT / f"burst_{src}"
    paths, ideal_idx = make_burst(src, out_dir, ideal_shot=ideal)

    fp = FramePick()
    result = fp.run_series(paths, series_name=f"burst_{src}")
    dest = fp.export(result, out_dir / "result")
    # 报告用白底版本（export 默认是深色终端风格）
    sheet = contact_sheet(result, dest / "contact_sheet_light.jpg", theme="light")

    top = result.ranking[0]
    gt = Path(paths[ideal_idx]).name
    ok = top.file == gt
    gap = float(result.ranking[0].utility - result.ranking[-1].utility)
    print(f"[图B] 组内 {len(paths)} 张，Top-1={top.file} 人工首选={gt} -> {'命中' if ok else '未命中'}"
          f" | U(top)={result.ranking[0].utility:.3f} U(last)={result.ranking[-1].utility:.3f} Δ={gap:.3f}")

    out = FIG / "real_group.jpg"
    Image.open(sheet).save(out, quality=90)
    return out


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    figure_candidates("p1074", top_show=4)
    figure_group("p1025", ideal=2)
