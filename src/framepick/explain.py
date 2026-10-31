"""把特征数值翻译成人话——选片结果的可解释性。

规则引擎 + 模板，好处是永远说得清"为什么选它"，也不依赖任何模型；
多模态大模型可用时，把它生成的 ``reason`` 拼在最前面，形成"模型判断 + 规则佐证"的双通道解释。
"""

from __future__ import annotations

__all__ = ["describe", "compose_reason"]

_TH_DEF = 1 / 3
_TH_HI = 2 / 3


def _position_word(feat: dict) -> str:
    t, c = feat.get("thirds", 0.0), feat.get("center", 0.0)
    if t >= 0.75:
        return "主体落在三分线交点附近"
    if c >= 0.75:
        return "主体居中、画面稳定"
    if t >= 0.5:
        return "主体位置接近三分构图"
    return "主体位置略偏，构图不够稳"


def _quality_words(feat: dict) -> list[str]:
    out: list[str] = []
    s, e = feat.get("sharpness", 0.0), feat.get("exposure", 0.0)
    if s >= 0.7:
        out.append("焦点锐利")
    elif s <= 0.35:
        out.append("画面偏软（有失焦/抖动风险）")
    if e >= 0.7:
        out.append("曝光正常")
    elif e <= 0.35:
        out.append("曝光偏差（过曝或欠曝）")
    if feat.get("clutter", 1.0) <= 0.4:
        out.append("边缘有干扰物")
    return out


def describe(feat: dict) -> str:
    parts = [_position_word(feat)]
    parts += _quality_words(feat)
    cov = feat.get("coverage", 0.0)
    if cov >= 0.9:
        parts.append("主体完整未被裁断")
    elif cov <= 0.7:
        parts.append("主体被裁掉一部分")
    if feat.get("fill", 0.0) >= 0.7:
        parts.append("主体占比适中、留白舒服")
    if feat.get("horizon", 1.0) <= 0.5:
        parts.append("地平线略有倾斜")
    if feat.get("colorfulness", 0.0) >= 0.75:
        parts.append("色彩通透")
    return "，".join(parts) + "。"


def compose_reason(feat: dict, vlm_reason: str | None = None, rank: int = 0) -> str:
    rule = describe(feat)
    if vlm_reason:
        return f"{vlm_reason.strip()}（规则佐证：{rule.rstrip('。')}）"
    return ("组内最优：" if rank == 0 else "") + rule
