"""端到端流水线：一组照片 → 排序 → 最佳照片 + 最佳取景框 + 理由。

设计要点：

* **组内相对化**：融合前先对整组（所有照片的所有候选框）做逐维标准化，
  直接对抗"相似图绝对分数趋同"的问题；
* **级联打分**：先用便宜的特征筛出每组 Top-N，再只对决赛选手调用美学模型 / 多模态大模型，
  在算力有限时这是最划算的取舍；
* **输出可复现**：所有中间量（候选数、各维分数、裁剪框）都写进 JSON，便于写报告和答辩。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .candidates import CropCandidate, generate_candidates
from .config import Config
from .explain import compose_reason
from .features import candidate_features
from .fusion import FusionWeights, fuse
from .io_utils import IMAGE_EXTS, dhash, hamming, load_image, save_image
from .saliency import saliency_map
from .scorers.base import HeuristicScorer

__all__ = ["FramePick", "PhotoResult", "SeriesResult", "CandidateResult"]


@dataclass
class CandidateResult:
    photo: str
    box: tuple[int, int, int, int]
    kind: str
    features: dict[str, float] = field(default_factory=dict)
    utility: float = 0.0

    def as_dict(self) -> dict:
        return {
            "photo": self.photo,
            "box": list(self.box),
            "kind": self.kind,
            "utility": round(self.utility, 4),
            "features": {k: round(float(v), 4) for k, v in self.features.items() if not k.startswith("raw_")},
        }


@dataclass
class PhotoResult:
    file: str
    path: str
    box: tuple[int, int, int, int]
    utility: float
    features: dict[str, float]
    reason: str = ""
    n_candidates: int = 0
    duplicate_of: str | None = None
    vlm: dict[str, float] | None = None
    scale: float = 1.0  # 工作图 → 原图的坐标缩放系数（crop / scale = 原图坐标）

    def scores(self) -> dict[str, float]:
        f = self.features
        return {
            "quality": round(float(f.get("quality", 0.0)), 3),
            "aesthetic": round(float(f.get("aesthetic", 0.0)), 3),
            "composition": round(float(f.get("composition", 0.0)), 3),
            "subject": round(float(f.get("subject", 0.0)), 3),
        }

    def as_dict(self) -> dict:
        return {
            "file": self.file,
            "crop": list(self.box),
            "utility": round(float(self.utility), 4),
            "scores": self.scores(),
            "reason": self.reason,
            "n_candidates": self.n_candidates,
            "duplicate_of": self.duplicate_of,
            "vlm": self.vlm,
        }


@dataclass
class SeriesResult:
    series: str
    best: PhotoResult
    ranking: list[PhotoResult]
    elapsed: float = 0.0
    config: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "series": self.series,
            "best": self.best.as_dict(),
            "ranking": [p.as_dict() for p in self.ranking],
            "elapsed_sec": round(self.elapsed, 2),
            "config": self.config,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


class FramePick:
    def __init__(
        self,
        config: Config | None = None,
        weights: FusionWeights | None = None,
        aesthetic_scorer=None,
        vlm_scorer=None,
    ):
        self.cfg = config or Config()
        self.weights = weights or FusionWeights()
        self.heuristic = HeuristicScorer()
        self.aesthetic_scorer = aesthetic_scorer
        self.vlm = vlm_scorer

    # ---------------------------------------------------------------- 单图
    def _photo_candidates(self, rgb: np.ndarray, name: str) -> list[CandidateResult]:
        sal = saliency_map(rgb)
        cands: list[CropCandidate] = generate_candidates(
            rgb,
            sal,
            grid=self.cfg.grid,
            max_candidates=self.cfg.max_candidates,
            min_side=self.cfg.min_side,
            dedup_iou=self.cfg.dedup_iou,
        )
        out: list[CandidateResult] = []
        for c in cands:
            feat = candidate_features(rgb, sal, c.box)
            # 深度美学模型（可选）只用于替换 heuristic 的美学分
            if self.aesthetic_scorer is not None and getattr(self.aesthetic_scorer, "available", lambda: False)():
                try:
                    sub = rgb[c.box[1] : c.box[3], c.box[0] : c.box[2]]
                    res = self.aesthetic_scorer.score(sub)
                    if "aesthetic" in res:
                        feat["aesthetic_model"] = feat["aesthetic"]
                        feat["aesthetic"] = float(res["aesthetic"])
                except Exception:
                    pass
            out.append(CandidateResult(photo=name, box=c.box, kind=c.kind, features=feat))
        return out

    # ---------------------------------------------------------------- 组
    def run_series(self, paths: Sequence[str | Path], series_name: str | None = None) -> SeriesResult:
        t0 = time.time()
        files = [Path(p) for p in paths if Path(p).suffix.lower() in IMAGE_EXTS]
        if not files:
            raise ValueError("没有找到可用的图片文件（支持 jpg/png/webp/tif 等）")
        files = sorted(files)
        name = series_name or files[0].parent.name

        loaded: list[tuple[str, np.ndarray, float, int]] = []
        for p in files:
            rgb, scale = load_image(p, max_side=self.cfg.work_size)
            loaded.append((p.name, rgb, scale, dhash(rgb)))
            if self.cfg.verbose:
                print(f"[load] {p.name} {rgb.shape[1]}x{rgb.shape[0]}")

        all_cands: list[CandidateResult] = []
        per_photo: dict[str, list[CandidateResult]] = {}
        for nm, rgb, _s, _h in loaded:
            cands = self._photo_candidates(rgb, nm)
            per_photo[nm] = cands
            all_cands.extend(cands)
            if self.cfg.verbose:
                print(f"[cand] {nm}: {len(cands)} 个候选取景框")

        # ---- 第一阶段：单图内部选最优取景框（绝对打分即可，不与其它照片混在一起）
        per_photo_best: dict[str, CandidateResult] = {}
        for nm, cands in per_photo.items():
            u = fuse([c.features for c in cands], self.weights, mode="absolute")
            for c, val in zip(cands, u):
                c.utility = float(val)
            per_photo_best[nm] = max(cands, key=lambda c: c.utility)

        # ---- 第二阶段：组内比较照片（相对化：只在这 N 张之间标准化）
        photo_list = [(nm, per_photo_best[nm]) for nm, _r, _s, _h in loaded]
        photo_utils = fuse(
            [c.features for _nm, c in photo_list],
            self.weights,
            mode=self.cfg.fusion_mode,
        )
        for (nm, c), u in zip(photo_list, photo_utils):
            c.utility = float(u)

        # 重复图标记
        dup: dict[str, str] = {}
        for i, (nm, _r, _s, h) in enumerate(loaded):
            for nm2, _r2, _s2, h2 in loaded[:i]:
                if hamming(h, h2) <= self.cfg.dedup_hamming:
                    dup[nm] = nm2
                    break

        photo_results: list[PhotoResult] = []
        scale_map = {nm: s for nm, _r, s, _h in loaded}
        for nm, rgb, _s, _h in loaded:
            best = per_photo_best[nm]
            photo_results.append(
                PhotoResult(
                    file=nm,
                    path=str(next(p for p in files if p.name == nm)),
                    box=best.box,
                    utility=best.utility,
                    features=best.features,
                    n_candidates=len(per_photo[nm]),
                    duplicate_of=dup.get(nm),
                    scale=_s,
                )
            )

        photo_results.sort(key=lambda p: p.utility, reverse=True)

        # 级联第二阶段：多模态大模型只对决赛选手复评
        if self.cfg.use_vlm and self.vlm is not None and getattr(self.vlm, "available", lambda: False)():
            photo_results = self._vlm_rerank(photo_results)

        for i, pr in enumerate(photo_results):
            vlm_reason = (pr.vlm or {}).get("reason", "") if pr.vlm else ""
            pr.reason = compose_reason(pr.features, str(vlm_reason) or None, rank=i)

        # 记录每个结果对应的原图坐标缩放系数
        for pr in photo_results:
            pr.scale = scale_map.get(pr.file, 1.0) or 1.0

        elapsed = time.time() - t0
        if self.cfg.verbose:
            print(f"[done] {name}: {len(photo_results)} 张，耗时 {elapsed:.1f}s，最佳 {photo_results[0].file}")

        return SeriesResult(
            series=name,
            best=photo_results[0],
            ranking=photo_results,
            elapsed=elapsed,
            config={"fusion_mode": self.cfg.fusion_mode, "max_candidates": self.cfg.max_candidates},
        )

    def _vlm_rerank(self, ranked: list[PhotoResult]) -> list[PhotoResult]:
        topn = max(1, min(self.cfg.vlm_topn, len(ranked)))
        w = self.cfg.vlm_weight
        head = ranked[:topn]
        for pr in head:
            try:
                img, _ = load_image(pr.path, max_side=self.cfg.work_size)
                x1, y1, x2, y2 = pr.box
                sub = img[y1:y2, x1:x2]
                res = self.vlm.score(sub)
                pr.vlm = {k: round(float(v), 3) for k, v in res.items() if isinstance(v, (int, float))}
                if "aesthetic" in res:
                    pr.utility = float((1 - w) * pr.utility + w * float(res["aesthetic"]))
            except Exception as exc:  # 接口失败不影响主流程
                pr.vlm = {"error": str(exc)[:60]}
        head.sort(key=lambda p: p.utility, reverse=True)
        return head + ranked[topn:]

    # ---------------------------------------------------------------- 导出
    def export(self, result: SeriesResult, out_dir: str | Path, save_crops: bool = True) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.to_json(out / "result.json")

        if not save_crops:
            return out

        crops_dir = out / "crops"
        crops_dir.mkdir(exist_ok=True)
        for i, pr in enumerate(result.ranking):
            img, _ = load_image(pr.path, max_side=self.cfg.work_size)
            x1, y1, x2, y2 = pr.box
            if i == 0 or i < self.cfg.topk:
                save_image(img[y1:y2, x1:x2], crops_dir / f"top{i + 1}_{Path(pr.file).stem}.jpg")
        return out
