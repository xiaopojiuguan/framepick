"""单元测试：python -m pytest tests -q   （或直接 python tests/test_pipeline.py）"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framepick.candidates import iou, generate_candidates
from framepick.features import candidate_features, technical_features
from framepick.fusion import FusionWeights, fuse, group_normalize
from framepick.ranker import BradleyTerry, ndcg_at_k, pairwise_accuracy
from framepick.saliency import saliency_map


def _fake_scene(h=240, w=360, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:] = np.array([0.55, 0.72, 0.92])
    img[int(h * 0.6) :] = np.array([0.42, 0.55, 0.35])
    # 一个高显著度的"主体"
    cy, cx = int(h * 0.68), int(w * 0.63)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy - cy) / 26) ** 2 + ((xx - cx) / 16) ** 2 <= 1
    img[mask] = np.array([0.12, 0.14, 0.18])
    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def test_iou_identity():
    a = (0, 0, 100, 100)
    assert abs(iou(a, a) - 1.0) < 1e-6
    assert iou(a, (100, 100, 200, 200)) == 0.0


def test_candidates_cover_salient_region():
    img = _fake_scene()
    sal = saliency_map(img)
    cands = generate_candidates(img, sal, max_candidates=24)
    assert len(cands) >= 5
    h, w = img.shape[:2]
    for c in cands:
        x1, y1, x2, y2 = c.box
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    # 显著区域应被至少一个候选框较好覆盖
    best = min(cands, key=lambda c: -float(sal[c.box[1] : c.box[3], c.box[0] : c.box[2]].mean()))
    assert best.area > 0


def test_technical_features_sanity():
    img = _fake_scene()
    t = technical_features(img)
    for k in ("sharpness", "exposure", "contrast", "noise", "entropy"):
        assert 0.0 <= t[k] <= 1.0
    blurred = img.copy()
    blurred[1:-1, 1:-1] = (
        img[:-2, 1:-1] + img[2:, 1:-1] + img[1:-1, :-2] + img[1:-1, 2:]
    ) / 4.0
    assert technical_features(blurred)["raw_lap_var"] < t["raw_lap_var"]


def test_composition_features_range():
    img = _fake_scene()
    sal = saliency_map(img)
    h, w = img.shape[:2]
    f = candidate_features(img, sal, (0, 0, w, h))
    assert 0.0 <= f["composition"] <= 1.0
    assert 0.0 <= f["coverage"] <= 1.0
    assert f["coverage"] > 0.99  # 整图必然覆盖全部显著性


def test_fuse_monotonic():
    feats = [
        {"quality": 0.9, "aesthetic": 0.9, "composition": 0.9, "subject": 0.9},
        {"quality": 0.5, "aesthetic": 0.5, "composition": 0.5, "subject": 0.5},
        {"quality": 0.2, "aesthetic": 0.2, "composition": 0.2, "subject": 0.2},
    ]
    u = fuse(feats, FusionWeights(), mode="relative")
    assert u[0] > u[1] > u[2]
    assert u.min() >= 0.0 and u.max() <= 1.0


def test_group_normalize_zero_mean():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Z = group_normalize(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-6)


def test_bradley_terry_recovers_order():
    rng = np.random.default_rng(0)
    true_w = np.array([1.0, 0.0])
    X = rng.normal(size=(12, 2))
    score = X @ true_w
    pairs = [(i, j) for i in range(12) for j in range(i + 1, 12) if score[i] > score[j]]
    m = BradleyTerry(l2=1e-4, iters=1500, lr=0.5).fit(X, pairs)
    pred = m.score(X)
    assert pairwise_accuracy(pred, pairs) > 0.95
    assert m.w[0] > 0.5


def test_ndcg_and_pairwise():
    scores = np.array([0.9, 0.6, 0.3])
    rel = np.array([3, 2, 1])
    assert ndcg_at_k(scores, rel, k=3) == 1.0
    assert pairwise_accuracy(scores, [(0, 1), (1, 2)]) == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"[pass] {fn.__name__}")
    print(f"\n{len(fns)} 项测试全部通过")
