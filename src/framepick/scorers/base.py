"""可插拔的评分器。

三类实现：

* ``HeuristicScorer``   —— 无需模型权重，用 features.py 的规则特征，保证任何环境都能跑通；
* ``NIMAScorer``        —— NIMA 分布回归（AVA 预训练），需 torch + 权重；
* ``CLIPAestheticScorer`` —— CLIP 特征 + 线性美学头（LAION-Aesthetic 风格），需 transformers；
* ``VLMScorer``         —— 多模态大模型评分 + 自然语言理由，走 OpenAI 兼容 HTTP 接口。

统一接口：``available() -> bool`` 与 ``score(rgb) -> dict``，其中必须含 ``aesthetic``（0~1）。
任何模型缺失时上层自动降级到启发式，不会中断流水线。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

__all__ = ["BaseScorer", "HeuristicScorer"]


class BaseScorer(Protocol):
    name: str

    def available(self) -> bool: ...

    def score(self, rgb: np.ndarray) -> dict[str, float]: ...


class HeuristicScorer:
    """规则特征打分，零依赖（除 numpy/PIL）。"""

    name = "heuristic"

    def available(self) -> bool:
        return True

    def score(self, rgb: np.ndarray) -> dict[str, float]:
        from ..features import technical_features

        t = technical_features(rgb)
        aesthetic = (
            0.34 * t["sharpness"]
            + 0.26 * t["exposure"]
            + 0.18 * t["contrast"]
            + 0.12 * t["colorfulness"]
            + 0.10 * t["entropy"]
        )
        return {"aesthetic": float(aesthetic), "sharpness": t["sharpness"]}
