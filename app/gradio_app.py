"""Gradio 演示界面：上传一组连拍 → 得到排名、最佳取景框和理由。

    pip install gradio
    python app/gradio_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framepick import Config, FramePick  # noqa: E402
from framepick.io_utils import save_image  # noqa: E402


def build():
    import gradio as gr
    from PIL import Image

    def run(files, topk, mode, use_vlm):
        if not files:
            return [], "请先上传同一场景的一组照片（建议 4~20 张）"
        tmp = Path(tempfile.mkdtemp(prefix="framepick_"))
        paths = []
        for i, f in enumerate(files):
            p = tmp / f"IMG_{i + 1:03d}.jpg"
            Image.open(f).convert("RGB").save(p, quality=95)
            paths.append(p)

        cfg = Config(topk=int(topk), fusion_mode=mode, use_vlm=bool(use_vlm), verbose=False)
        vlm = None
        if cfg.use_vlm:
            from framepick.scorers.vlm import VLMScorer

            vlm = VLMScorer()
        fp = FramePick(cfg, vlm=vlm)
        result = fp.run_series(paths, series_name="upload")

        gallery = []
        for i, pr in enumerate(result.ranking[: int(topk)]):
            arr = np.asarray(Image.open(pr.path).convert("RGB"), dtype=np.float32) / 255.0
            x1, y1, x2, y2 = pr.box
            crop_path = tmp / f"rank{i + 1}.jpg"
            save_image(arr[y1:y2, x1:x2], crop_path)
            gallery.append((str(crop_path), f"#{i + 1} {pr.file}  util={pr.utility:.3f}"))

        s = result.best.scores()
        md = [
            f"### 🏆 最佳：`{result.best.file}`",
            f"**取景框** `{result.best.box}`　**效用** `{result.best.utility:.3f}`",
            "",
            "| 质量 | 美学 | 构图 | 主体 |",
            "| --- | --- | --- | --- |",
            f"| {s['quality']:.2f} | {s['aesthetic']:.2f} | {s['composition']:.2f} | {s['subject']:.2f} |",
            "",
            f"**理由**：{result.best.reason}",
            "",
            "| 名次 | 文件 | 效用 |",
            "| --- | --- | --- |",
        ]
        md += [f"| {i + 1} | {p.file} | {p.utility:.3f} |" for i, p in enumerate(result.ranking)]
        return gallery, "\n".join(md)

    with gr.Blocks(title="FramePick 智能选片") as demo:
        gr.Markdown("# FramePick · 智能选片\n上传同一场景的一组照片，自动选出最值得保留的一张并给出最佳取景框。")
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(file_count="multiple", label="一组同场景照片", type="filepath")
                topk = gr.Slider(1, 6, value=3, step=1, label="展示前 K 名")
                mode = gr.Radio(["relative", "absolute"], value="relative", label="融合模式")
                vlm = gr.Checkbox(value=False, label="启用多模态大模型复评（需 API Key）")
                btn = gr.Button("开始选片", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="排名与取景框", columns=3, height=420)
                info = gr.Markdown()
        btn.click(run, [files, topk, mode, vlm], [gallery, info])
    return demo


if __name__ == "__main__":
    try:
        build().launch()
    except ImportError:
        print("未安装 gradio。运行：pip install gradio\n或改用命令行：python -m framepick.cli --input <dir>")
