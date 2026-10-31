"""NIMA：Neural Image Assessment (Talebi & Milanfar, TIP 2018)。

把美学回归变成**分布预测**：网络输出 10 个 bin 上的概率（打分 1~10），
期望值即美学分，方差反映"人们意见的分歧程度"——对选片很有用：
分歧小的图（构图干净）通常比分歧大的图更稳妥。

依赖：torch + torchvision。权重缺省放在 ``weights/nima_ava.pth``，
也支持 ``torch.hub`` 拉取社区复现权重。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["NIMAScorer"]


class NIMAScorer:
    name = "nima"

    def __init__(self, weights: str | Path | None = None, device: str | None = None, input_size: int = 224):
        self.weights_path = Path(weights) if weights else Path("weights/nima_ava.pth")
        self.input_size = input_size
        self.device = device
        self._model = None
        self._torch = None

    # ---------------------------------------------------------------- 加载
    def available(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch  # noqa: F401
        except Exception:
            return False
        if not self.weights_path.exists():
            return False
        try:
            self._load()
            return True
        except Exception:
            return False

    def _load(self) -> None:
        import torch
        from torchvision import models

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.last_channel, 10)
        state = torch.load(str(self.weights_path), map_location="cpu")
        state = state.get("state_dict", state)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self.device).eval()
        self._torch = torch

    # ---------------------------------------------------------------- 推理
    def score(self, rgb: np.ndarray) -> dict[str, float]:
        if not self.available():
            raise RuntimeError("NIMA 不可用：缺少 torch 或权重文件")
        torch = self._torch
        from PIL import Image

        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        img = img.resize((self.input_size, self.input_size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        x = torch.from_numpy(arr.transpose(2, 0, 1)[None]).to(self.device)

        with torch.no_grad():
            logits = self._model(x)[0]
            p = torch.softmax(logits, dim=-1).cpu().numpy()

        bins = np.arange(1, 11, dtype=np.float32)
        mean_score = float((p * bins).sum())
        std_score = float(np.sqrt(((bins - mean_score) ** 2 * p).sum()))
        return {
            "aesthetic": float(np.clip((mean_score - 1.0) / 9.0, 0.0, 1.0)),
            "nima_mean": mean_score,
            "nima_std": std_score,
            "nima_consensus": float(np.clip(1.0 - std_score / 2.5, 0.0, 1.0)),
        }
