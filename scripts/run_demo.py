"""一键跑通演示：生成数据（如缺失）→ 跑流水线 → 出拼图 → 打印排序表。

    python scripts/run_demo.py                 # 用 data/sample/series_01
    python scripts/run_demo.py --all           # 跑所有系列并汇总 Top-1 命中率
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framepick import Config, FramePick  # noqa: E402
from framepick.io_utils import IMAGE_EXTS  # noqa: E402
from framepick.visualize import contact_sheet  # noqa: E402


def run_one(series_dir: Path, out_root: Path, cfg: Config) -> dict:
    paths = sorted(p for p in series_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    fp = FramePick(cfg)
    result = fp.run_series(paths, series_name=series_dir.name)
    out = fp.export(result, out_root / series_dir.name)
    sheet = contact_sheet(result, out / "contact_sheet.jpg")

    gt_file = series_dir / "ground_truth.json"
    gt = json.loads(gt_file.read_text(encoding="utf-8"))["best"] if gt_file.exists() else None
    hit = (result.best.file == gt) if gt else None

    print(f"\n[{series_dir.name}] 最佳 = {result.best.file}   人工首选 = {gt}   {'✅' if hit else ('❌' if hit is False else '—')}")
    print(f"  {'名次':<4}{'文件':<14}{'效用':>8}  分项（质量/美学/构图）")
    for i, pr in enumerate(result.ranking, 1):
        s = pr.scores()
        print(f"  {i:<4}{pr.file:<14}{pr.utility:>8.3f}  {s['quality']:.2f} / {s['aesthetic']:.2f} / {s['composition']:.2f}")
    print(f"  理由: {result.best.reason}")
    print(f"  拼图: {sheet}")
    return {"series": series_dir.name, "best": result.best.file, "gt": gt, "hit": hit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/sample/series_01"))
    ap.add_argument("--out", default=str(ROOT / "outputs/demo"))
    ap.add_argument("--all", action="store_true", help="跑 data/sample 下所有系列")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    cfg = Config(topk=args.topk)
    out_root = Path(args.out)

    if args.all:
        base = Path(args.input).parent
        series_dirs = sorted(d for d in base.iterdir() if d.is_dir())
        rows = [run_one(d, out_root, cfg) for d in series_dirs]
        hits = [r["hit"] for r in rows if r["hit"] is not None]
        if hits:
            print(f"\n=== 汇总：Top-1 命中率 {sum(hits)}/{len(hits)} = {sum(hits) / len(hits):.1%} ===")
    else:
        run_one(Path(args.input), out_root, cfg)


if __name__ == "__main__":
    main()
