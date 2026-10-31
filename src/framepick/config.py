"""全局配置。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["Config"]


@dataclass
class Config:
    # --- 预处理
    work_size: int = 1024          # 长边归一化尺寸（特征计算在这个尺度上做）
    min_side: int = 96             # 候选框最小边长（工作图尺度）

    # --- 候选生成
    max_candidates: int = 32       # 每张图保留的候选数，直接影响耗时
    grid: int = 4                  # 网格滑窗密度
    dedup_iou: float = 0.88        # 候选去重的 IoU 阈值

    # --- 融合与排序
    fusion_mode: str = "relative"  # relative | absolute
    topk: int = 3

    # --- 多模态大模型（级联第二阶段，只对决赛选手调用，省钱）
    use_vlm: bool = False
    vlm_topn: int = 3
    vlm_weight: float = 0.25

    # --- 重复图检测
    dedup_hamming: int = 6

    verbose: bool = True
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
