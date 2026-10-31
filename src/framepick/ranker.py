"""偏好排序学习（Learning to Rank）。

本项目的最终目标不是"给一张图打分"，而是"把一组图排对"。所以训练信号用
**组内成对偏好**（A 比 B 好），模型用 Bradley-Terry / RankNet 的对数似然：

    P(A ≻ B) = σ(w·(φ(A) − φ(B)))

损失：L = −Σ log P(A ≻ B) + λ‖w‖²

同时给出论文里常用的评价指标：成对准确率、Top-1 命中率、NDCG@k、Kendall τ。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "BradleyTerry",
    "pairwise_accuracy",
    "top1_accuracy",
    "ndcg_at_k",
    "kendall_tau",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class BradleyTerry:
    l2: float = 1e-3
    lr: float = 0.35
    iters: int = 600
    lr_decay: float = 0.995
    w: np.ndarray | None = None

    def fit(self, X: np.ndarray, pairs: list[tuple[int, int]]) -> "BradleyTerry":
        """X: (n, d) 特征；pairs: [(i, j)] 表示第 i 个优于第 j 个。"""
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        w = np.zeros(d, dtype=np.float64)
        if not pairs:
            self.w = w
            return self

        idx_a = np.asarray([p[0] for p in pairs], dtype=int)
        idx_b = np.asarray([p[1] for p in pairs], dtype=int)
        diff = X[idx_a] - X[idx_b]

        lr = self.lr
        for _ in range(self.iters):
            p = _sigmoid(diff @ w)
            grad = diff.T @ (1.0 - p) / len(pairs) - 2.0 * self.l2 * w
            w += lr * grad
            lr *= self.lr_decay
        self.w = w
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("尚未训练：请先调用 fit() 或 load()")
        return 1.0 / (1.0 + np.exp(-np.asarray(X, dtype=np.float64) @ self.w))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"w": (self.w or np.zeros(0)).tolist(), "l2": self.l2}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BradleyTerry":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(l2=float(d.get("l2", 1e-3)))
        m.w = np.asarray(d["w"], dtype=np.float64)
        return m


# ---------------------------------------------------------------- 评价指标


def pairwise_accuracy(scores: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    ok = sum(1.0 for i, j in pairs if scores[i] > scores[j])
    return ok / len(pairs)


def top1_accuracy(series_scores: list[np.ndarray], series_gt: list[int]) -> float:
    """每组取分数最高的那张，是否等于人工首选。"""
    if not series_scores:
        return float("nan")
    hits = sum(1 for s, g in zip(series_scores, series_gt) if int(np.argmax(s)) == g)
    return hits / len(series_scores)


def ndcg_at_k(scores: np.ndarray, rel: np.ndarray, k: int = 3) -> float:
    """rel 为人工相关度（例如 3/2/1/0 的最佳/次佳/一般/淘汰）。"""
    order = np.argsort(-scores)[:k]
    gains = (2.0 ** rel[order] - 1.0) / np.log2(np.arange(2, len(order) + 2))
    ideal_idx = np.argsort(-rel)[:k]
    ideal = (2.0 ** rel[ideal_idx] - 1.0) / np.log2(np.arange(2, len(ideal_idx) + 2))
    denom = ideal.sum()
    return float(gains.sum() / denom) if denom > 0 else float("nan")


def kendall_tau(a: list[int], b: list[int]) -> float:
    """两个排序（元素为 id 序列）的 Kendall τ。"""
    pos = {v: i for i, v in enumerate(b)}
    seq = [pos[v] for v in a if v in pos]
    n = len(seq)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (0.5 * n * (n - 1))
