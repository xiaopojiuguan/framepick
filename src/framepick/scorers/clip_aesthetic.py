"""CLIP / CLIP-IQA 风格的美学打分。

两条路：

1. **线性美学头**（LAION-Aesthetic / AVA 上训练的 linear probe）：CLIP 图像特征 → 1 维美学分。
   需要 ``weights/clip_aesthetic.npz``（含 W、b、均值方差）。
2. **CLIP-IQA 提示词法**（Wang et al., 2023）：用 "Good photo." / "Bad photo." 两个文本嵌入的
   相似度差做无参考质量分，不需要额外训练，但分辨率敏感、易受内容分布影响。

本项目把两者都封装进来，默认优先线性头（更贴美学），缺失时退回提示词法。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["CLIPAestheticScorer"]

GOOD_PROMPTS = ("a high quality photo", "a beautiful photograph", "good composition, sharp, well exposed")
BAD_PROMPTS = ("a low quality photo", "a blurry badly composed snapshot", "poorly exposed noisy photo")


class CLIPAestheticScorer:
    name = "clip_aesthetic"

    def __init__(self, head: str | Path | None = None, model_name: str = "openai/clip-vit-base-patch32"):
        self.head_path = Path(head) if head else Path("weights/clip_aesthetic.npz")
        self.model_name = model_name
        self._model = None
        self._proc = None
        self._torch = None
        self._head = None

    def available(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        try:
            self._load()
            return True
        except Exception:
            return False

    def _load(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(self.model_name).eval()
        self._proc = CLIPProcessor.from_pretrained(self.model_name)
        self._torch = torch
        if self.head_path.exists():
            d = np.load(self.head_path)
            self._head = {k: d[k] for k in d.files}

    def _embed(self, rgb: np.ndarray):
        from PIL import Image

        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        inputs = self._proc(images=img, return_tensors="pt")
        with self._torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0]

    def score(self, rgb: np.ndarray) -> dict[str, float]:
        if not self.available():
            raise RuntimeError("CLIP 不可用：缺少 torch/transformers 或本地权重的网络访问")
        f = self._embed(rgb)
        if self._head is not None:
            mu, sd = self._head["mu"], self._head["sd"]
            z = (f - mu) / np.maximum(sd, 1e-6)
            val = float(z @ self._head["W"] + float(self._head["b"][0]))
            s = 1.0 / (1.0 + np.exp(-val))
            return {"aesthetic": float(np.clip(s, 0, 1)), "clip_linear": val}

        # 提示词法
        texts = list(GOOD_PROMPTS) + list(BAD_PROMPTS)
        inputs = self._proc(text=texts, return_tensors="pt", padding=True)
        with self._torch.no_grad():
            tf = self._model.get_text_features(**inputs)
        tf = (tf / tf.norm(dim=-1, keepdim=True)).cpu().numpy()
        sims = tf @ f
        k = len(GOOD_PROMPTS)
        good, bad = float(sims[:k].mean()), float(sims[k:].mean())
        s = 1.0 / (1.0 + np.exp(-(good - bad) * 100.0))
        return {"aesthetic": float(np.clip(s, 0, 1)), "clip_gap": good - bad}
