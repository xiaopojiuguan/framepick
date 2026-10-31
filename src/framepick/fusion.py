"""特征融合：把多路评分压成单张照片（或单个取景框）的效用值。

两条可切换的融合方式：

* ``absolute``：直接对归一化后的子分数加权（0~1 之间的语义化分数），可解释性好，
  但同组相似照片之间差异会被压缩。
* ``relative``：先做**组内标准化**（每条维度减去本组均值、除以本组标准差），再加权。
  这正是 Chang et al. (2016) 指出的关键——同场景连拍时绝对分数失效，相对分数才有区分度。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["FusionWeights", "fuse", "group_normalize"]

# 参与融合的维度（与 features.FEATURE_NAMES 的前 4 项 + semantic 对应）
BASE_DIMS: tuple[str, ...] = ("quality", "aesthetic", "composition", "subject")


@dataclass
class FusionWeights:
    quality: float = 0.30
    aesthetic: float = 0.26
    composition: float = 0.30
    subject: float = 0.14
    semantic: float = 0.16
    bias: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FusionWeights":
        known = {k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def load(cls, path: str | Path) -> "FusionWeights":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def vector(self, dims: tuple[str, ...]) -> np.ndarray:
        return np.asarray([float(getattr(self, d, 0.0)) for d in dims], dtype=np.float64)


def group_normalize(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """按列（特征维）做组内 z-score。"""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / np.maximum(sd, eps)


def _sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fuse(
    feats: list[dict[str, float]],
    weights: FusionWeights | None = None,
    mode: str = "relative",
    semantic: list[float] | None = None,
) -> np.ndarray:
    """把一组候选的特征字典融合成 [0,1] 的效用值数组。

    参数
    ----
    feats : 每个候选取景框的特征字典列表
    weights : 融合权重，缺省使用设计初值
    mode : ``relative``（组内标准化，推荐）或 ``absolute``
    semantic : 可选的多模态大模型语义分（与 feats 等长），缺失时按权重重新分配
    """
    w = weights or FusionWeights()
    dims = list(BASE_DIMS)
    has_sem = semantic is not None
    if has_sem:
        dims.append("semantic")

    X = np.asarray([[float(f.get(d, 0.0)) for d in dims] for f in feats], dtype=np.float64)
    if has_sem:
        X[:, -1] = np.asarray(semantic, dtype=np.float64).reshape(-1)

    if mode == "relative" and len(feats) > 2:
        X = group_normalize(X)
        # 组内标准化后落在 [-3,3]，压到 [0,1] 便于加权解释
        X = _sigmoid(X)
    else:
        X = np.clip(X, 0.0, 1.0)

    wv = w.vector(tuple(dims))
    wv = wv / (wv.sum() + 1e-9)
    return _sigmoid(X @ wv * 4.0 - 2.0 + w.bias)  # 拉伸到 [0,1] 且保留排序
