"""谱残差显著性检测 (Spectral Residual Saliency).

实现 Hou & Zhang, "Saliency Detection: A Spectral Residual Approach", CVPR 2007。
纯 numpy 实现，无 OpenCV 依赖，用于给候选取景框提供"主体在哪"的先验。

对一张图做：
  1. 缩放到 64x64 灰度；
  2. FFT 取对数幅度谱，用均值滤波后的幅度谱做残差 —— 抑制统计上冗余的背景；
  3. 用残差 + 原相位重建，得到空域的显著性响应；
  4. 双边上采样回原尺寸并归一化。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = ["saliency_map", "saliency_bbox", "saliency_peaks"]


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB float [0,1] -> 灰度 float [0,1]。"""
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _resize(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    img = img.resize((hw[1], hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def box_blur(img: np.ndarray, k: int = 3) -> np.ndarray:
    """积分图实现的均值滤波，边界按 edge 复制。"""
    k = max(1, int(k) | 1)  # 保证奇数
    pad = k // 2
    p = np.pad(img, pad, mode="edge")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    h, w = img.shape
    out = c[k : k + h, k : k + w] - c[0:h, k : k + w] - c[k : k + h, 0:w] + c[0:h, 0:w]
    return (out / float(k * k)).astype(np.float32)


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def saliency_map(rgb: np.ndarray, base: int = 64, blur: int = 9) -> np.ndarray:
    """返回与原图同尺寸、取值 [0,1] 的显著性图。"""
    gray = to_gray(rgb)
    small = _resize(gray, (base, base))

    spec = np.fft.fft2(small)
    log_amp = np.log(np.abs(spec) + 1e-8)
    phase = np.angle(spec)

    residual = log_amp - box_blur(log_amp, 3)
    sal_small = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal_small = box_blur(sal_small.astype(np.float32), blur)
    sal_small = _norm01(sal_small)

    h, w = gray.shape
    return _resize(sal_small, (h, w))


def saliency_bbox(sal: np.ndarray, thresh_ratio: float = 0.78) -> tuple[int, int, int, int]:
    """主体包围盒。

    用**相对阈值**（相对峰值）而不是绝对阈值：谱残差图的绝对幅度随图像内容变化很大，
    固定 0.55 之类的阈值在强纹理图上会把整幅图都框进去。0.78×峰值经验上对应
    "最显著的一块主体"。
    """
    h, w = sal.shape
    mask = sal >= thresh_ratio * float(sal.max())
    if mask.sum() < max(16, 0.0015 * h * w):
        cy, cx = np.unravel_index(int(np.argmax(sal)), sal.shape)
        half_h, half_w = h // 4, w // 4
        return (max(0, cx - half_w), max(0, cy - half_h), min(w, cx + half_w), min(h, cy + half_h))
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def saliency_peaks(sal: np.ndarray, k: int = 3, min_dist_ratio: float = 0.25) -> list[tuple[int, int, float]]:
    """返回若干显著峰 (x, y, value)，用于多点关注的主体布局判断。"""
    h, w = sal.shape
    work = sal.copy()
    r = max(3, int(min(h, w) * min_dist_ratio) // 2)
    peaks: list[tuple[int, int, float]] = []
    for _ in range(k):
        idx = int(np.argmax(work))
        cy, cx = np.unravel_index(idx, work.shape)
        val = float(work[cy, cx])
        if val <= 0.15:
            break
        peaks.append((int(cx), int(cy), val))
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        work[y0:y1, x0:x1] = 0.0
    return peaks
