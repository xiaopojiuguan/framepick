"""评分器集合。所有实现都遵循 ``available()`` + ``score(rgb) -> dict`` 的约定。"""

from __future__ import annotations

from .base import BaseScorer, HeuristicScorer

__all__ = ["BaseScorer", "HeuristicScorer", "NIMAScorer", "CLIPAestheticScorer", "VLMScorer"]


def __getattr__(name: str):
    # 惰性导入，避免没有 torch / 没有网络时 import 就炸
    if name == "NIMAScorer":
        from .nima import NIMAScorer

        return NIMAScorer
    if name == "CLIPAestheticScorer":
        from .clip_aesthetic import CLIPAestheticScorer

        return CLIPAestheticScorer
    if name == "VLMScorer":
        from .vlm import VLMScorer

        return VLMScorer
    raise AttributeError(name)
