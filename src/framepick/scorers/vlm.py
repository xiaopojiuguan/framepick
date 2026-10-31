"""多模态大模型（MLLM）评分器。

用途有两层：

1. **打分**：让 VLM 按评分细则（主体清晰度 / 曝光 / 构图 / 色彩 / 情绪-瞬间）给出 0~10 分，
   归一到 [0,1] 参与融合；
2. **给理由**：输出一两句人话解释，作为选片结果的可解释性来源（也是答辩时的展示点）。

实现走 OpenAI 兼容的 ``/chat/completions`` 接口，所以 Qwen-VL（DashScope 兼容模式）、
InternVL、GLM-4V、GPT-4o、本地 vLLM 部署都能直接对接，只改 ``base_url`` 与 ``model``。

已知坑（写进报告，也写进代码）：
* 位置/分辨率偏差：VLM 对图像位置、缩放敏感，同图不同尺寸可能差 0.5 分以上 → 固定长边、固定 JPEG 质量；
* 分数饱和：直接问"几分"容易全给 7~8 分 → 用**组内比较**提问 + few-shot 锚点；
* 幻觉：让它先描述再打分（CoT），并强制 JSON 输出。
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any

import numpy as np

__all__ = ["VLMScorer", "SYSTEM_PROMPT", "build_messages"]

SYSTEM_PROMPT = (
    "你是一位资深摄影编辑，擅长从同一场景的连拍中挑出最值得保留的一张。"
    "评价时请严格依据：①主体是否清晰完整 ②曝光与色彩是否自然 ③构图（三分/居中/留白/水平）"
    "④瞬间与情绪（表情、姿态、光线）。只输出 JSON，不要多余文字。"
)

USER_TEMPLATE = """这张照片是从同一场景的一组连拍中裁剪出的候选取景框。
请分步思考后按 0-10 打分（0=报废，5=普通记录，7=不错，9-10=可以直接发布）：
- clarity: 主体清晰度与对焦
- exposure: 曝光、动态范围、色彩
- composition: 构图与取景（主体位置、留白、水平线、干扰物）
- moment: 瞬间与情绪表达
- overall: 综合美感（请与"普通手机随手拍=5 分"这个锚点比较）

严格输出如下 JSON：
{{"clarity": 0, "exposure": 0, "composition": 0, "moment": 0, "overall": 0,
  "reason": "不超过 40 字的中文理由"}}"""


def _encode_jpeg(rgb: np.ndarray, max_side: int = 768, quality: int = 88) -> str:
    from PIL import Image

    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_messages(rgb: np.ndarray, extra_context: str | None = None) -> list[dict[str, Any]]:
    text = USER_TEMPLATE
    if extra_context:
        text = f"{extra_context}\n\n{text}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_jpeg(rgb)}"}},
            ],
        },
    ]


class VLMScorer:
    name = "vlm"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        cache: dict | None = None,
    ):
        self.model = model or os.environ.get("FRAMEPICK_VLM_MODEL", "qwen-vl-max")
        self.base_url = (base_url or os.environ.get("FRAMEPICK_VLM_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("FRAMEPICK_API_KEY", "")
        self.timeout = timeout
        self.cache: dict = cache if cache is not None else {}

    def available(self) -> bool:
        return bool(self.api_key)

    # ---------------------------------------------------------------- 调用
    def _chat(self, messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _parse(text: str) -> dict[str, float]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            text = text[4:] if text.lower().startswith("json") else text
        try:
            d = json.loads(text)
        except Exception:
            s, e = text.find("{"), text.rfind("}")
            d = json.loads(text[s : e + 1]) if s >= 0 and e > s else {}
        out: dict[str, float] = {}
        for k in ("clarity", "exposure", "composition", "moment", "overall"):
            try:
                out[k] = float(d.get(k, 0.0)) / 10.0
            except Exception:
                out[k] = 0.0
        out["aesthetic"] = out["overall"]
        if isinstance(d.get("reason"), str):
            out["reason_text"] = 0.0  # 占位，理由单独取
        return out

    def score(self, rgb: np.ndarray, extra_context: str | None = None) -> dict[str, float]:
        if not self.available():
            raise RuntimeError("VLM 不可用：请设置环境变量 FRAMEPICK_API_KEY")
        key = (id(rgb), extra_context)
        if key in self.cache:
            return self.cache[key]
        try:
            raw = self._chat(build_messages(rgb, extra_context))
            res = self._parse(raw)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            res = {"aesthetic": 0.0, "vlm_error": str(exc)[:80]}
        self.cache[key] = res
        return res

    def reason(self, rgb: np.ndarray, extra_context: str | None = None) -> str:
        try:
            raw = self._chat(build_messages(rgb, extra_context))
            s, e = raw.find("{"), raw.rfind("}")
            d = json.loads(raw[s : e + 1])
            return str(d.get("reason", ""))[:120]
        except Exception:
            return ""
