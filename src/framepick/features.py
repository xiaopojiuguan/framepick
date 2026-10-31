"""候选画框的特征提取：技术质量 / 构图 / 显著性。

全部使用 numpy + PIL 实现，便于在没有 GPU、没有 OpenCV 的环境里跑通原型；
深度美学模型（NIMA / CLIP-IQA）在 ``framepick.scorers`` 中以适配器形式挂载，
输出同样归一到 [0,1] 后与本模块的特征融合。

约定：图像统一为 float32、形状 (H, W, 3)、取值 [0,1]。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .saliency import to_gray

__all__ = [
    "Box",
    "technical_features",
    "composition_features",
    "candidate_features",
    "FEATURE_NAMES",
]

Box = tuple[int, int, int, int]  # x1, y1, x2, y2


# ---------------------------------------------------------------- 基础工具


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _laplacian(gray: np.ndarray) -> np.ndarray:
    p = np.pad(gray, 1, mode="edge")
    return (
        -4.0 * p[1:-1, 1:-1]
        + p[:-2, 1:-1]
        + p[2:, 1:-1]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
    )


def _gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(gray, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx.astype(np.float32), gy.astype(np.float32)


def _entropy(gray: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(1, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / math.log2(bins))  # 归一到 [0,1]


def _noise_sigma(gray: np.ndarray) -> float:
    """Immerkær (1996) 快速噪声估计。"""
    p = np.pad(gray, 1, mode="edge")
    m = (
        p[:-2, :-2]
        - 2 * p[:-2, 1:-1]
        + p[:-2, 2:]
        - 2 * p[1:-1, :-2]
        + 4 * p[1:-1, 1:-1]
        - 2 * p[1:-1, 2:]
        + p[2:, :-2]
        - 2 * p[2:, 1:-1]
        + p[2:, 2:]
    )
    return float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(m)) / 6.0)


def _colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colorfulness，经验值范围 0~100+。"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean()))


def crop(rgb: np.ndarray, box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = rgb.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return rgb[y1:y2, x1:x2]


# ---------------------------------------------------------------- 技术质量


def technical_features(rgb: np.ndarray) -> dict[str, float]:
    """对一张（已裁剪的）图给出 6 个技术质量指标，均归一到 [0,1]。"""
    gray = to_gray(rgb) if rgb.ndim == 3 else rgb
    small = gray
    if max(gray.shape) > 512:  # 控制计算量
        step = int(math.ceil(max(gray.shape) / 512))
        small = gray[::step, ::step]

    lap = _laplacian(small)
    lap_var = float(lap.var())

    mean_lum = float(gray.mean())
    hi = float((gray > 0.98).mean())
    lo = float((gray < 0.02).mean())
    std_lum = float(gray.std())

    sharpness = _clip01(1.0 - math.exp(-lap_var / 0.0015)) if lap_var < 5 else 1.0
    # 曝光：理想均值 ~0.45，两端溢出惩罚
    exposure = _clip01(math.exp(-((mean_lum - 0.45) / 0.30) ** 2) - 2.0 * (hi + lo))
    contrast = _clip01(1.0 - math.exp(-std_lum / 0.18))
    colorfulness = _clip01(_colorfulness(rgb) / 65.0) if rgb.ndim == 3 else 0.5
    noise = _clip01(1.0 - _noise_sigma(small) / 0.06)
    entropy = _clip01(_entropy(small))

    return {
        "sharpness": sharpness,
        "exposure": exposure,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "noise": noise,
        "entropy": entropy,
        "raw_lap_var": lap_var,
        "raw_mean_lum": mean_lum,
        "raw_clip_high": hi,
        "raw_clip_low": lo,
    }


# ---------------------------------------------------------------- 构图


def _horizon_tilt(gray: np.ndarray) -> float:
    """估计地平线倾斜角（度，0 表示完全水平）。无地平线时返回 0。"""
    if max(gray.shape) > 480:
        step = int(math.ceil(max(gray.shape) / 480))
        gray = gray[::step, ::step]
    gx, gy = _gradients(gray)
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx))  # 梯度方向
    mask = (np.abs(ang) > 75.0) & (mag > np.percentile(mag, 80))
    if mask.sum() < 50:
        return 0.0
    tilt = 90.0 - np.abs(ang[mask])
    hist, edges = np.histogram(tilt, bins=37, range=(0.0, 18.0), weights=mag[mask])
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2.0)


def _thirds_score(cx: float, cy: float, w: float, h: float) -> float:
    """主体位置与最近三分点的距离 → 分数（1 表示落在三分点上）。"""
    nodes = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
    d = min(math.hypot(cx / w - nx, cy / h - ny) for nx, ny in nodes)
    return _clip01(1.0 - d / 0.28)


def _center_score(cx: float, cy: float, w: float, h: float) -> float:
    d = math.hypot(cx / w - 0.5, cy / h - 0.5)
    return _clip01(1.0 - d / 0.35)


def composition_features(rgb: np.ndarray, sal: np.ndarray, box: Box) -> dict[str, float]:
    """针对某个候选框计算构图相关特征。"""
    x1, y1, x2, y2 = box
    sub = crop(rgb, box)
    if sub.size == 0:
        return {"composition": 0.0}
    h, w = sub.shape[:2]
    sal_c = crop(sal[..., None], box)[..., 0] if sal.ndim == 2 else crop(sal, box)

    gray = to_gray(sub)
    # 主体完整性：整图显著区域的包围盒有多少比例落在候选框内（1 = 主体没被裁断）
    from .saliency import saliency_bbox

    gx1, gy1, gx2, gy2 = saliency_bbox(sal if sal.ndim == 2 else sal[..., 0])
    inter_w = max(0, min(x2, gx2) - max(x1, gx1))
    inter_h = max(0, min(y2, gy2) - max(y1, gy1))
    subj_area = max(1, (gx2 - gx1) * (gy2 - gy1))
    coverage = _clip01(inter_w * inter_h / float(subj_area))

    # 主体质心（加权）
    ys, xs = np.mgrid[0:h, 0:w]
    wgt = sal_c + 1e-6
    cx = float((wgt * xs).sum() / wgt.sum())
    cy = float((wgt * ys).sum() / wgt.sum())

    thirds = _thirds_score(cx, cy, w, h)
    center = _center_score(cx, cy, w, h)

    # 主体包围盒与画框的留白（避免切头切脚）
    m = sal_c > 0.6
    if m.sum() > 0:
        sy, sx = np.where(m)
        bx1, bx2, by1, by2 = int(sx.min()), int(sx.max()) + 1, int(sy.min()), int(sy.max()) + 1
        margin_l, margin_r = bx1, w - bx2
        margin_t, margin_b = by1, h - by2
        margin = min(margin_l, margin_r, margin_t, margin_b) / float(min(w, h))
        fill = ((bx2 - bx1) * (by2 - by1)) / float(w * h)
    else:
        margin, fill = 0.0, coverage
    # 留白过小 → 压抑；留白过大 → 主体太小
    margin_score = _clip01(1.0 - abs(margin - 0.10) / 0.18)
    fill_score = _clip01(1.0 - abs(fill - 0.22) / 0.30)

    # 左右/上下显著性平衡
    left, right = float(sal_c[:, : max(1, w // 2)].sum()), float(sal_c[:, w // 2 :].sum())
    top, bottom = float(sal_c[: max(1, h // 2), :].sum()), float(sal_c[h // 2 :, :].sum())
    balance = _clip01(1.0 - abs(left - right) / (left + right + 1e-8))
    vbalance = _clip01(1.0 - abs(top - bottom) / (top + bottom + 1e-8))

    # 对称性
    q = max(1, w // 2)
    diff = float(np.abs(gray[:, :q] - gray[:, -q:][:, ::-1]).mean())
    symmetry = _clip01(1.0 - diff / 0.22)

    # 边缘杂乱度：外框 8% 区域的梯度能量占比
    gx, gy = _gradients(gray)
    edge = np.hypot(gx, gy)
    b = max(1, int(min(w, h) * 0.08))
    border = np.zeros_like(edge, dtype=bool)
    border[:b, :] = border[-b:, :] = True
    border[:, :b] = border[:, -b:] = True
    clutter = _clip01(1.0 - float(edge[border].mean()) / (float(edge.mean()) + 1e-6) / 3.0)

    tilt = _horizon_tilt(gray)
    horiz = _clip01(1.0 - tilt / 8.0)

    parts = {
        "coverage": coverage,
        "thirds": thirds,
        "center": center,
        "margin": margin_score,
        "fill": fill_score,
        "balance": balance,
        "vbalance": vbalance,
        "symmetry": symmetry,
        "clutter": clutter,
        "horizon": horiz,
    }
    # 构图总分：三分/居中取较优，其余加权
    parts["composition"] = _clip01(
        0.26 * coverage
        + 0.24 * max(thirds, 0.85 * center)
        + 0.14 * margin_score
        + 0.10 * fill_score
        + 0.10 * balance
        + 0.08 * symmetry
        + 0.08 * clutter
    ) * _clip01(0.85 + 0.15 * horiz)
    parts["raw_tilt_deg"] = tilt
    parts["raw_fill"] = fill
    parts["raw_margin"] = margin
    return parts


def candidate_features(rgb_full: np.ndarray, sal_full: np.ndarray, box: Box) -> dict[str, float]:
    """单张照片的某个候选框的完整特征。"""
    sub = crop(rgb_full, box)
    tech = technical_features(sub)
    comp = composition_features(rgb_full, sal_full, box)

    quality = _clip01(
        0.30 * tech["sharpness"]
        + 0.24 * tech["exposure"]
        + 0.16 * tech["contrast"]
        + 0.10 * tech["colorfulness"]
        + 0.10 * tech["noise"]
        + 0.10 * tech["entropy"]
    )

    sal_c = crop(sal_full, box) if sal_full.ndim == 2 else crop(sal_full[..., 0], box)
    subject = _clip01(float(sal_c.mean()) / 0.22)  # 显著性平均强度，约 0.22 视为"主体明确"

    feat = {**tech, **comp}
    feat["quality"] = quality
    feat["subject"] = subject
    feat["aesthetic"] = _clip01(
        0.42 * quality + 0.30 * comp["composition"] + 0.18 * tech["colorfulness"] + 0.10 * subject
    )
    return feat


FEATURE_NAMES: list[str] = [
    "quality",
    "aesthetic",
    "composition",
    "subject",
    "coverage",
    "thirds",
    "center",
    "margin",
    "fill",
    "balance",
    "symmetry",
    "clutter",
    "sharpness",
    "exposure",
    "contrast",
    "colorfulness",
    "noise",
]


def as_vector(feat: dict[str, float]) -> np.ndarray:
    return np.asarray([float(feat.get(k, 0.0)) for k in FEATURE_NAMES], dtype=np.float32)
