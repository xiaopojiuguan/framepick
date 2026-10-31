"""用组内人工偏好训练排序器（Bradley-Terry / RankNet 损失）。

数据格式 ``labels.csv``（``tools/label_tool.py`` 直接产出）：

    series,file,rank
    series_01,IMG_003.jpg,1
    series_01,IMG_001.jpg,2
    ...

流程：
1. 对每张图枚举候选取景框，取"参考分（美学+构图启发式）"最高的 K 个候选，特征求均值，
   作为**图片级特征**——注意这一步与待学习的融合权重无关，所以不会自我循环；
2. 同组内按 rank 生成成对偏好（rank 小者胜）；
3. 最小化 −Σ log σ(w·Δφ) + λ‖w‖²；
4. 留一组交叉验证，报告成对准确率与 Top-1 命中率。

    python scripts/train_ranker.py --labels data/labels.csv --out weights/fusion_bt.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framepick.candidates import generate_candidates  # noqa: E402
from framepick.features import FEATURE_NAMES, candidate_features  # noqa: E402
from framepick.io_utils import load_image  # noqa: E402
from framepick.ranker import BradleyTerry, pairwise_accuracy, top1_accuracy  # noqa: E402
from framepick.saliency import saliency_map  # noqa: E402

TOPK_PER_PHOTO = 5


def photo_feature(path: Path, cache: dict) -> np.ndarray:
    key = str(path)
    if key in cache:
        return np.asarray(cache[key], dtype=np.float64)
    rgb, _ = load_image(path, max_side=1024)
    sal = saliency_map(rgb)
    cands = generate_candidates(rgb, sal, max_candidates=32)
    feats = [candidate_features(rgb, sal, c.box) for c in cands]
    feats.sort(key=lambda f: f["aesthetic"] + f["composition"], reverse=True)
    feats = feats[:TOPK_PER_PHOTO]
    vec = np.mean([[f[k] for k in FEATURE_NAMES] for f in feats], axis=0)
    cache[key] = vec.tolist()
    return vec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "data/labels.csv"))
    ap.add_argument("--out", default=str(ROOT / "weights/fusion_bt.json"))
    ap.add_argument("--l2", type=float, default=5e-3)
    ap.add_argument("--iters", type=int, default=800)
    args = ap.parse_args()

    label_path = Path(args.labels)
    if not label_path.exists():
        print(f"找不到标注文件 {label_path}，先跑 tools/label_tool.py 完成标注")
        return

    rows: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with label_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["series"]].append((r["file"], int(r["rank"])))

    cache_path = ROOT / "outputs/feature_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    series_list, X_all, y_all = [], [], []
    for s, items in sorted(rows.items()):
        items.sort(key=lambda t: t[1])
        vecs = []
        for fname, _rank in items:
            p = ROOT / "data" / "series" / s / fname
            if not p.exists():
                p = Path(fname)
            vecs.append(photo_feature(p, cache))
        X_all.append(np.stack(vecs))
        series_list.append((s, items))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    # 留一组交叉验证
    accs, top1s = [], []
    for hold in range(len(X_all)):
        X_tr = np.concatenate([X_all[i] for i in range(len(X_all)) if i != hold], axis=0)
        pairs_tr: list[tuple[int, int]] = []
        off = 0
        for i in range(len(X_all)):
            if i == hold:
                continue
            n = X_all[i].shape[0]
            pairs_tr += [(off + a, off + b) for a in range(n) for b in range(a + 1, n)]
            off += n
        model = BradleyTerry(l2=args.l2, iters=args.iters).fit(X_tr, pairs_tr)

        X_te = X_all[hold]
        n = X_te.shape[0]
        pairs_te = [(a, b) for a in range(n) for b in range(a + 1, n)]
        scores = model.score(X_te)
        accs.append(pairwise_accuracy(scores, pairs_te))
        top1s.append(1.0 if int(np.argmax(scores)) == 0 else 0.0)

    print(f"留一组交叉验证：成对准确率 {np.nanmean(accs):.3f}    Top-1 {np.nanmean(top1s):.3f}")

    # 全量重训并保存
    X = np.concatenate(X_all, axis=0)
    pairs: list[tuple[int, int]] = []
    off = 0
    for i, (_s, items) in enumerate(series_list):
        n = X_all[i].shape[0]
        pairs += [(off + a, off + b) for a in range(n) for b in range(a + 1, n)]
        off += n
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Xn = (X - mu) / np.maximum(sd, 1e-6)
    model = BradleyTerry(l2=args.l2, iters=args.iters).fit(Xn, pairs)

    out = {
        "feature_names": FEATURE_NAMES,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "w": (model.w or np.zeros(0)).tolist(),
        "note": "先标准化再线性打分；Top-1 高的维度是选片时最该关注的视觉因子",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    order = np.argsort(-np.abs(model.w))[:8]
    print("\n权重绝对值 Top-8：")
    for k in order:
        print(f"  {FEATURE_NAMES[k]:<14} {model.w[k]:+.4f}")
    print(f"\n已保存 {args.out}")


if __name__ == "__main__":
    main()
