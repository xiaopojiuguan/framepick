"""FramePick —— 组内智能选片与取景框优化。

    from framepick import FramePick, Config

    fp = FramePick(Config(max_candidates=32))
    result = fp.run_series(["a.jpg", "b.jpg", "c.jpg"])
    print(result.best.file, result.best.reason)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .candidates import CropCandidate, generate_candidates, iou
from .config import Config
from .features import candidate_features, technical_features, composition_features
from .fusion import FusionWeights, fuse, group_normalize
from .pipeline import FramePick, SeriesResult, PhotoResult
from .saliency import saliency_map

__all__ = [
    "__version__",
    "FramePick",
    "Config",
    "SeriesResult",
    "PhotoResult",
    "CropCandidate",
    "generate_candidates",
    "iou",
    "candidate_features",
    "technical_features",
    "composition_features",
    "saliency_map",
    "FusionWeights",
    "fuse",
    "group_normalize",
]
