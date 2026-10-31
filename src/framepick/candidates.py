"""候选取景框生成。

思路：单张照片上不是任意裁剪都合理，人的取景有强烈先验。因此候选由三路合成：

1. **整图基线**：原图本身（相当于"不裁剪"）。
2. **网格滑窗**：在若干宽高比 × 若干面积比例的格点上滑动，保证召回。
3. **构图锚定框**：把显著性主体分别锚定在三分点 / 中心点，反解窗口位置，
   保证"好构图"的框一定在候选集里（先验注入，而不是全靠评价器学出来）。

三路合并后用显著性质量 + 构图先验排序，再做 IoU 抑制，保留 Top-K。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["CropCandidate", "generate_candidates", "iou", "DEFAULT_ASPECTS"]

Box = tuple[int, int, int, int]

DEFAULT_ASPECTS: tuple[float, ...] = (1.0, 4 / 3, 3 / 4, 3 / 2, 2 / 3, 16 / 9, 9 / 16)
DEFAULT_SCALES: tuple[float, ...] = (0.55, 0.70, 0.85, 1.0)

# 构图锚点（相对窗口的归一化位置）
ANCHORS: tuple[tuple[float, float, str], ...] = (
    (1 / 3, 1 / 3, "thirds-lt"),
    (2 / 3, 1 / 3, "thirds-rt"),
    (1 / 3, 2 / 3, "thirds-lb"),
    (2 / 3, 2 / 3, "thirds-rb"),
    (0.5, 0.5, "center"),
)


@dataclass(frozen=True)
class CropCandidate:
    box: Box
    aspect: float
    scale: float
    kind: str
    prior: float = 0.0

    @property
    def area(self) -> int:
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])

    def as_dict(self) -> dict:
        return {
            "box": list(self.box),
            "aspect": round(float(self.aspect), 3),
            "scale": round(float(self.scale), 3),
            "kind": self.kind,
            "prior": round(float(self.prior), 4),
        }


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / max(1, ua))


def _integral(x: np.ndarray) -> np.ndarray:
    s = x.cumsum(0).cumsum(1)
    out = np.zeros((x.shape[0] + 1, x.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = s
    return out


def _rect_sum(ii: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1])


def _thirds_prior(cx: float, cy: float, w: float, h: float) -> float:
    nodes = ((1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3), (0.5, 0.5))
    d = min(math.hypot(cx / w - nx, cy / h - ny) for nx, ny in nodes)
    return float(max(0.0, 1.0 - d / 0.35))


def generate_candidates(
    rgb: np.ndarray,
    sal: np.ndarray,
    aspects: tuple[float, ...] = DEFAULT_ASPECTS,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    grid: int = 4,
    max_candidates: int = 48,
    min_side: int = 96,
    dedup_iou: float = 0.88,
) -> list[CropCandidate]:
    """给单张照片生成候选取景框。"""
    h, w = rgb.shape[:2]
    ii = _integral(np.asarray(sal, dtype=np.float64))
    total_mass = float(ii[-1, -1]) + 1e-9

    # 显著性质心
    ys, xs = np.mgrid[0:h, 0:w]
    wgt = np.asarray(sal, dtype=np.float64) + 1e-6
    scx = float((wgt * xs).sum() / wgt.sum())
    scy = float((wgt * ys).sum() / wgt.sum())

    cands: list[CropCandidate] = []

    # 1) 整图
    cands.append(CropCandidate((0, 0, w, h), w / h, 1.0, "full", prior=1.0))

    def add(box: Box, aspect: float, scale: float, kind: str) -> None:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, min(w - 1, int(round(x1)))), max(0, min(h - 1, int(round(y1))))
        x2, y2 = max(x1 + min_side, min(w, int(round(x2)))), max(y1 + min_side, min(h, int(round(y2))))
        if x2 - x1 < min_side or y2 - y1 < min_side:
            return
        area = (x2 - x1) * (y2 - y1)
        mass = _rect_sum(ii, x1, y1, x2, y2) / total_mass
        area_ratio = area / float(w * h)
        # 面积先验：太小丢信息，太大等于没裁
        area_prior = math.exp(-((math.sqrt(area_ratio) - 0.72) / 0.35) ** 2)
        pos = _thirds_prior(scx * 1.0, scy * 1.0, w, h)
        prior = 0.62 * mass + 0.26 * pos + 0.12 * area_prior
        cands.append(CropCandidate((x1, y1, x2, y2), aspect, scale, kind, prior))

    for a in aspects:
        # 该宽高比下能画出的最大框
        if a >= w / h:
            max_w, max_h = float(w), w / a
        else:
            max_w, max_h = h * a, float(h)
        for s in scales:
            bw, bh = max_w * s, max_h * s
            if bw < min_side or bh < min_side:
                continue
            # 2) 网格滑窗
            nx = max(1, grid)
            for ix in range(nx + 1):
                for iy in range(nx + 1):
                    x1 = (w - bw) * ix / nx
                    y1 = (h - bh) * iy / nx
                    add((x1, y1, x1 + bw, y1 + bh), a, s, "grid")
            # 3) 构图锚定
            for ax, ay, name in ANCHORS:
                x1 = scx - ax * bw
                y1 = scy - ay * bh
                add((x1, y1, x1 + bw, y1 + bh), a, s, name)

    # 按先验排序 + 去重
    cands.sort(key=lambda c: c.prior, reverse=True)
    kept: list[CropCandidate] = []
    for c in cands:
        if all(iou(c.box, k.box) < dedup_iou for k in kept):
            kept.append(c)
        if len(kept) >= max_candidates:
            break
    return kept
