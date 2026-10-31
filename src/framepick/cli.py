"""命令行入口。

    python -m framepick.cli --input data/sample/series_01 --out outputs/run01 --topk 3 --sheet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .fusion import FusionWeights
from .io_utils import IMAGE_EXTS
from .pipeline import FramePick
from .visualize import contact_sheet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="framepick", description="智能选片：从一组同场景照片中选出最佳一张")
    ap.add_argument("--input", required=True, help="一组照片所在目录，或若干图片路径")
    ap.add_argument("--out", default="outputs/run", help="输出目录")
    ap.add_argument("--config", default=None, help="配置文件（JSON）")
    ap.add_argument("--weights", default=None, help="融合权重（JSON）")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--mode", choices=["relative", "absolute"], default=None)
    ap.add_argument("--vlm", action="store_true", help="启用多模态大模型复评（需 FRAMEPICK_API_KEY）")
    ap.add_argument("--sheet", action="store_true", help="额外输出对比拼图")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    cfg.topk = args.topk
    if args.mode:
        cfg.fusion_mode = args.mode
    cfg.use_vlm = bool(args.vlm)
    cfg.verbose = not args.quiet

    weights = FusionWeights.load(args.weights) if args.weights else None

    paths: list[Path] = []
    for token in args.input.split(","):
        p = Path(token.strip())
        if p.is_dir():
            paths.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS))
        elif p.exists():
            paths.append(p)

    vlm = None
    if cfg.use_vlm:
        from .scorers.vlm import VLMScorer

        vlm = VLMScorer()
        if not vlm.available():
            print("[warn] 未检测到 FRAMEPICK_API_KEY，已跳过 VLM 复评", file=sys.stderr)

    fp = FramePick(config=cfg, weights=weights, vlm=vlm)
    result = fp.run_series(paths)
    out = fp.export(result, args.out)

    if args.sheet:
        sheet = contact_sheet(result, out / "contact_sheet.jpg")
        print(f"[sheet] {sheet}")

    best = result.best
    print("\n=== 最佳照片 ===")
    print(f"文件: {best.file}")
    print(f"取景框: {best.box}  效用: {best.utility:.4f}")
    print(f"分项: {best.scores()}")
    print(f"理由: {best.reason}")
    print(f"\n完整结果: {out / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
