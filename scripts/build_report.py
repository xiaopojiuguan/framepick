"""把 docs/设计报告.md 编译成可直接提交的 PDF。

为什么不用 Word：报告里有大量表格和公式，Markdown 是单一事实来源（GitHub 上也能直接看），
用 Chrome 无头模式打印成 PDF 能保证「网页看到什么、PDF 就是什么」，且不需要装任何 Office。

    python scripts/build_report.py
    → docs/设计报告.html （中间产物，可单独分享）
    → docs/设计报告.pdf  （提交用）
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SRC = DOCS / "设计报告.md"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]


# ---------------------------------------------------------------- 架构图（内联 SVG）


def architecture_svg() -> str:
    nodes = [
        (16, "一组同场景照片", "#1f2937"),
        (112, "预处理：EXIF 校正 / 长边归一化 / 去重", "#1f2937"),
        (208, "候选取景框生成（三路合成 + NMS）", "#1f2937"),
        (304, "候选评价：多路打分器", "#1f2937"),
        (400, "组内相对化 + 加权融合", "#b45309"),
        (496, "Bradley-Terry / 偏好排序", "#b45309"),
        (592, "输出：最佳照片 + 最佳取景框 + 理由", "#065f46"),
    ]
    side3 = ["① 整图基线（不裁剪）", "② 网格滑窗：7 种宽高比 × 4 档面积", "③ 构图锚定：三分点 / 中心反解"]
    side4 = ["技术质量：锐度 / 曝光 / 噪声 / 色彩", "美学模型：NIMA / CLIP-IQA", "构图规则：三分 / 留白 / 平衡 / 水平", "多模态大模型：语义与瞬间"]

    p = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="676" viewBox="0 0 720 676" font-family="Microsoft YaHei, PingFang SC, sans-serif">',
        '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L8,3 z" fill="#4b5563"/></marker></defs>',
        '<rect x="0" y="0" width="720" height="676" fill="#ffffff"/>',
    ]
    for y, label, color in nodes:
        p.append(
            f'<rect x="40" y="{y}" width="270" height="52" rx="9" fill="#f8fafc" stroke="{color}" stroke-width="1.4"/>'
        )
        p.append(
            f'<text x="175" y="{y + 30}" text-anchor="middle" font-size="13.5" fill="{color}">{html.escape(label)}</text>'
        )
    for i in range(len(nodes) - 1):
        y1 = nodes[i][0] + 52
        y2 = nodes[i + 1][0]
        p.append(f'<line x1="175" y1="{y1}" x2="175" y2="{y2 - 4}" stroke="#4b5563" stroke-width="1.4" marker-end="url(#ar)"/>')

    for i, t in enumerate(side3):
        y = 166 + i * 38
        p.append(f'<rect x="352" y="{y}" width="330" height="28" rx="6" fill="#eef6ff" stroke="#93c5fd"/>')
        p.append(f'<text x="364" y="{y + 19}" font-size="12" fill="#1e3a8a">{html.escape(t)}</text>')
    p.append('<path d="M310,234 C332,234 332,180 352,180" stroke="#93c5fd" stroke-width="1.2" fill="none" stroke-dasharray="4 3"/>')

    for i, t in enumerate(side4):
        y = 292 + i * 38
        p.append(f'<rect x="352" y="{y}" width="330" height="28" rx="6" fill="#f5f3ff" stroke="#c4b5fd"/>')
        p.append(f'<text x="364" y="{y + 19}" font-size="12" fill="#4c1d95">{html.escape(t)}</text>')
    p.append('<path d="M310,330 C332,330 332,306 352,306" stroke="#c4b5fd" stroke-width="1.2" fill="none" stroke-dasharray="4 3"/>')

    p.append(
        '<text x="352" y="474" font-size="11.5" fill="#6b7280">第一阶段：图内选最优框（绝对）'
        "</text>"
    )
    p.append('<text x="352" y="492" font-size="11.5" fill="#6b7280">第二阶段：组内 z-score 后比较（相对）</text>')
    p.append(
        '<rect x="348" y="458" width="252" height="46" rx="6" fill="none" stroke="#e5e7eb" stroke-dasharray="4 3"/>'
    )
    p.append("</svg>")
    return "\n".join(p)


# ---------------------------------------------------------------- 简易 LaTeX → HTML

SYMBOLS = {
    "dots": "…", "ldots": "…", "cdots": "⋯", "cdot": "·", "times": "×", "div": "÷",
    "pm": "±", "approx": "≈", "neq": "≠", "le": "≤", "ge": "≥", "ll": "≪", "gg": "≫",
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "cup": "∪", "cap": "∩",
    "sum": "Σ", "prod": "∏", "int": "∫", "infty": "∞", "partial": "∂", "nabla": "∇",
    "sigma": "σ", "Sigma": "Σ", "phi": "φ", "Phi": "Φ", "lambda": "λ", "Lambda": "Λ",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "mu": "μ",
    "tau": "τ", "pi": "π", "theta": "θ", "omega": "ω", "Omega": "Ω", "epsilon": "ε",
    "succ": "≻", "prec": "≺", "sim": "∼", "propto": "∝", "forall": "∀", "exists": "∃",
    "top": "ᵀ", "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒",
    "log": "log", "exp": "exp", "max": "max", "min": "min", "arg": "arg", "det": "det",
    "mathcal": "", "mathrm": "", "text": "", "left": "", "right": "", "quad": " ", "qquad": "  ",
    "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉", "mid": "|", "star": "*",
}


def latex_to_html(src: str) -> str:
    """把报告里用到的少量 LaTeX 转成 HTML（够用即可，不追求完备）。"""
    s = src.strip()
    # 文本类命令
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathcal\{([^{}]*)\}", r"\1", s)
    # 重音与根号
    s = re.sub(r"\\hat\{([^{}]*)\}", lambda m: m.group(1) + "\u0302", s)
    s = re.sub(r"\\bar\{([^{}]*)\}", lambda m: m.group(1) + "\u0304", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
    # 分数（支持 \frac{a}{b} / \frac12 两种写法）；先做上下标，才能让嵌套分数被正确匹配
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"\^\\?\*", "<sup>*</sup>", s)
    for _ in range(4):
        s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\[dt]?frac(\w)(\w)", r"(\1)/(\2)", s)
    # 花括号字面量（集合记号），先转实体避免后面被当成分组括号清掉
    s = s.replace("\\{", "&#123;").replace("\\}", "&#125;")
    # 下标/上标的简写形式
    s = re.sub(r"_(\w)", r"<sub>\1</sub>", s)
    s = re.sub(r"\^(\w)", r"<sup>\1</sup>", s)
    # 细空格与负空格
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\:", " ").replace("\\!", "")
    s = s.replace("\\ ", " ")

    def cmd(m: re.Match) -> str:
        return SYMBOLS.get(m.group(1), "")

    s = re.sub(r"\\([A-Za-z]+)", cmd, s)
    s = s.replace("\\|", "‖").replace("{", "").replace("}", "").replace("\\", "")
    s = s.replace("^ᵀ", "<sup>ᵀ</sup>")
    return re.sub(r"[ \t]{2,}", " ", s).strip()


# ---------------------------------------------------------------- Markdown 预处理


def preprocess(md_text: str) -> str:
    # 1) 抽掉 mermaid 块，换成内联 SVG
    md_text = re.sub(r"```mermaid.*?```", "@@ARCH@@", md_text, flags=re.S)

    # 2) 块级公式 $$...$$ → 带样式的 div
    def block(m: re.Match) -> str:
        return f'<div class="formula">{latex_to_html(m.group(1))}</div>'

    md_text = re.sub(r"\$\$(.+?)\$\$", block, md_text, flags=re.S)

    # 3) 行内公式 $...$ → <span class="imath">
    def inline(m: re.Match) -> str:
        return f'<span class="imath">{latex_to_html(m.group(1))}</span>'

    md_text = re.sub(r"\$([^$\n]{1,160})\$", inline, md_text)
    return md_text


COVER = """
<section class="cover">
  <div class="cover-tag">机器视觉 · 2026 期中设计报告</div>
  <h1 class="cover-title">智能选片</h1>
  <div class="cover-sub">从一组相似照片中选出最值得保留的一张<br/>——组内相对排序与取景框优化</div>
  <div class="cover-line"></div>
  <table class="cover-table">
    <tr><th>项目名称</th><td>FramePick · 智能选片与取景框优化系统</td></tr>
    <tr><th>课程</th><td>机器视觉（2026 秋季）</td></tr>
    <tr><th>组长</th><td>蒋锦宏　202493006</td></tr>
    <tr><th>组员</th><td>张满　202493002</td></tr>
    <tr><th>组员</th><td>章丞轩　202493001</td></tr>
    <tr><th>代码仓库</th><td>github.com/xiaopojiuguan/framepick</td></tr>
    <tr><th>提交日期</th><td>2026 年 11 月</td></tr>
  </table>
</section>
"""

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
@media screen {
  body { width: 210mm; margin: 0 auto; padding: 18mm 16mm; background: #fff; }
}
* { box-sizing: border-box; }
body {
  font-family: "Source Han Serif SC", "Songti SC", "SimSun", "Microsoft YaHei", serif;
  font-size: 10.4pt; line-height: 1.82; color: #1b1b1b; margin: 0;
  -webkit-font-smoothing: antialiased;
}
.cover { page-break-after: always; break-after: page; padding-top: 46mm; }
.cover-tag { font-family: "Microsoft YaHei", sans-serif; font-size: 10pt; letter-spacing: 3px;
  color: #2b6cb0; border-left: 3px solid #2b6cb0; padding-left: 10px; }
.cover-title { font-family: "Microsoft YaHei", sans-serif; font-size: 40pt; margin: 24mm 0 6mm;
  letter-spacing: 6px; font-weight: 700; color: #111827; }
.cover-sub { font-size: 13pt; line-height: 1.9; color: #374151; }
.cover-line { height: 2px; background: linear-gradient(90deg,#2b6cb0,#b45309); margin: 16mm 0 10mm; }
.cover-table { width: 100%; border-collapse: collapse; font-size: 10.5pt; }
.cover-table th { width: 26%; text-align: left; background: none; border: none;
  border-bottom: 1px solid #e5e7eb; color: #6b7280; font-weight: 400; padding: 7px 0; }
.cover-table td { border-bottom: 1px solid #e5e7eb; padding: 7px 0; }
.page-break { page-break-after: always; }

h1 { font-family: "Microsoft YaHei", sans-serif; font-size: 19pt; color: #111827;
  margin: 0 0 12px; padding-bottom: 8px; border-bottom: 2px solid #111827; }
h2 { font-family: "Microsoft YaHei", sans-serif; font-size: 14pt; color: #111827;
  margin: 26px 0 10px; padding-left: 9px; border-left: 4px solid #2b6cb0;
  page-break-after: avoid; }
h3 { font-family: "Microsoft YaHei", sans-serif; font-size: 11.6pt; color: #1f2937;
  margin: 18px 0 6px; page-break-after: avoid; }
h4 { font-family: "Microsoft YaHei", sans-serif; font-size: 10.6pt; margin: 14px 0 4px; }
p { margin: 7px 0; text-align: justify; }
ul, ol { margin: 7px 0 7px 4px; padding-left: 20px; }
li { margin: 3px 0; }
strong { color: #111827; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 18px 0; }
a { color: #2b6cb0; text-decoration: none; }

table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 9.1pt;
  font-family: "Microsoft YaHei", "Songti SC", sans-serif; page-break-inside: auto; }
th { background: #eef2f7; color: #1f2937; font-weight: 600; text-align: left;
  padding: 6px 8px; border: 1px solid #d5dde7; }
td { padding: 6px 8px; border: 1px solid #e3e8ef; vertical-align: top; }
tr { page-break-inside: avoid; }
tbody tr:nth-child(even) { background: #fafbfd; }

code { font-family: "Cascadia Mono", Consolas, monospace; font-size: 9.2pt;
  background: #f3f4f6; padding: 1px 4px; border-radius: 3px; color: #b91c1c; }
pre { background: #0f172a; color: #e5e7eb; padding: 12px 14px; border-radius: 8px;
  font-size: 8.8pt; line-height: 1.55; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; color: inherit; padding: 0; font-size: 8.8pt; }

.formula { font-family: "Cambria Math", "Times New Roman", serif; font-size: 12pt;
  text-align: center; margin: 12px 0; padding: 10px; background: #f8fafc;
  border-left: 3px solid #cbd5e1; color: #111827; }
.formula sub, .formula sup { font-size: 70%; }
.imath { font-family: "Cambria Math", "Times New Roman", serif; font-size: 10.6pt;
  color: #0f172a; padding: 0 1px; }
.imath sub, .imath sup { font-size: 70%; }

blockquote { margin: 10px 0; padding: 9px 14px; background: #fffbeb;
  border-left: 3px solid #f59e0b; color: #4b5563; font-size: 9.8pt; }

svg { display: block; margin: 14px auto; width: 100%; max-width: 100%; height: auto; }
img { max-width: 100%; height: auto; }
"""


def build() -> Path:
    md = SRC.read_text(encoding="utf-8")
    arch = architecture_svg()
    (DOCS / "figures").mkdir(exist_ok=True)
    (DOCS / "figures" / "architecture.svg").write_text(arch, encoding="utf-8")

    body_md = md.split("\n---\n", 1)[1] if md.startswith("#") is False else md
    # 去掉报告 Markdown 里自带的标题块（前 5 行元信息），封面已包含
    lines = md.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("## 摘要"))
    body_md = "\n".join(lines[start:])

    body_md = preprocess(body_md)
    body_html = markdown.markdown(body_md, extensions=["tables", "fenced_code", "sane_lists"])
    body_html = body_html.replace("@@ARCH@@", arch)

    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>智能选片 · 期中设计报告</title>
<style>{CSS}</style></head>
<body>{COVER}{body_html}</body></html>"""
    out_html = DOCS / "设计报告.html"
    out_html.write_text(doc, encoding="utf-8")
    return out_html


def to_pdf(html_path: Path) -> Path | None:
    chrome = next((c for c in CHROME_CANDIDATES if Path(c).exists()), None)
    if chrome is None:
        print("[warn] 未找到 Chrome/Edge，跳过 PDF（HTML 已生成，可手动打印为 PDF）")
        return None
    out_pdf = html_path.with_suffix(".pdf")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--virtual-time-budget=8000",
        f"--print-to-pdf={out_pdf}",
        html_path.as_uri(),
    ]
    print("[pdf]", " ".join(cmd[:5]), "...")
    subprocess.run(cmd, check=False, capture_output=True, timeout=180)
    return out_pdf if out_pdf.exists() else None


if __name__ == "__main__":
    h = build()
    print(f"[html] {h}")
    pdf = to_pdf(h)
    if pdf:
        print(f"[pdf ] {pdf}  ({pdf.stat().st_size / 1024:.0f} KB)")
    else:
        sys.exit(1)
