"""组内偏好标注小工具（零依赖，标准库 http.server）。

用法：

    python tools/label_tool.py --data data/series --labels data/labels.csv

浏览器打开 http://127.0.0.1:8765 ，左侧选一组，按"从左到右 = 从最好到最差"点击照片，
点"提交"即写入 ``labels.csv``（格式：series,file,rank）。已标注的组会打勾。

为什么不用 Label Studio：这个任务只需要"一组内排个序"，自己写 60 行前端比部署
一套标注平台快得多，而且产出格式直接就是本项目训练脚本吃的格式。
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>FramePick 标注台</title>
<style>
 :root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#e8eaee;--sub:#9aa3b2;--acc:#ffc23c;}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;display:flex;height:100vh}
 aside{width:260px;border-right:1px solid var(--line);padding:16px;overflow:auto}
 aside h1{font-size:15px;margin:0 0 12px;letter-spacing:.5px}
 aside button{display:flex;justify-content:space-between;width:100%;margin:4px 0;padding:9px 10px;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:8px;cursor:pointer;text-align:left}
 aside button:hover{border-color:var(--acc)} aside button.done{opacity:.55}
 main{flex:1;display:flex;flex-direction:column;padding:18px;overflow:auto}
 .hint{color:var(--sub);margin-bottom:10px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}
 .card{position:relative;border:2px solid var(--line);border-radius:10px;overflow:hidden;cursor:pointer;background:var(--panel)}
 .card img{width:100%;height:150px;object-fit:cover;display:block}
 .card .idx{position:absolute;top:6px;left:6px;background:#000a;border-radius:6px;padding:1px 8px;font-weight:700}
 .card.sel{border-color:var(--acc);box-shadow:0 0 0 2px #ffc23c33}
 .card select{width:100%;background:#12151b;color:var(--fg);border:0;border-top:1px solid var(--line);padding:6px}
 footer{padding:12px 0;display:flex;gap:12px;align-items:center}
 button.primary{background:var(--acc);color:#20180a;border:0;border-radius:8px;padding:10px 22px;font-weight:700;cursor:pointer}
 button.ghost{background:transparent;color:var(--sub);border:1px solid var(--line);border-radius:8px;padding:10px 16px;cursor:pointer}
 .ok{color:#7ee787}
</style></head>
<body>
<aside><h1>FramePick 标注台</h1><div id="list"></div></aside>
<main>
  <div class="hint" id="hint">选择左侧任意一组开始标注：给每张照片选一个名次（1 = 最值得保留）。</div>
  <div class="grid" id="grid"></div>
  <footer><button class="primary" id="submit" disabled>提交本组</button>
  <button class="ghost" id="clear">清空选择</button><span id="msg"></span></footer>
</main>
<script>
let cur=null, seriesList=[], ranks={};
const $=s=>document.querySelector(s);
async function refresh(){
  seriesList = await (await fetch('/api/series')).json();
  $('#list').innerHTML = seriesList.map(s=>
    `<button data-s="${s.name}" class="${s.done?'done':''}"><span>${s.name}</span><span>${s.done?'✓ '+s.n:'—'}</span></button>`).join('');
  document.querySelectorAll('#list button').forEach(b=>b.onclick=()=>open(b.dataset.s));
}
async function open(name){
  cur=name; ranks={};
  const d = await (await fetch('/api/series/'+encodeURIComponent(name))).json();
  $('#hint').textContent = `${name} · 共 ${d.files.length} 张 · 为每张选择名次，1 = 最值得保留`;
  $('#grid').innerHTML = d.files.map((f,i)=>`
    <div class="card" data-f="${f}">
      <img src="/img/${encodeURIComponent(name)}/${encodeURIComponent(f)}" loading="lazy"/>
      <div class="idx">?</div>
      <select><option value="">—</option>${d.files.map((_,k)=>`<option value="${k+1}">${k+1}</option>`).join('')}</select>
    </div>`).join('');
  document.querySelectorAll('.card').forEach(c=>{
    c.querySelector('select').onchange=e=>{
      const v=e.target.value; ranks[c.dataset.f]=v?parseInt(v):null;
      c.classList.toggle('sel', !!v); c.querySelector('.idx').textContent=v||'?'; sync();
    };
  });
  $('#submit').disabled=false; $('#msg').textContent='';
}
function sync(){ $('#submit').disabled = Object.values(ranks).filter(Boolean).length < 2; }
$('#clear').onclick=()=>{ranks={}; if(cur) open(cur);};
$('#submit').onclick=async()=>{
  const items=Object.entries(ranks).filter(([k,v])=>v);
  const res=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({series:cur, items})});
  const j=await res.json();
  $('#msg').innerHTML = j.ok? `<span class="ok">已保存 ${j.n} 条到 ${j.file}</span>` : '保存失败: '+j.error;
  refresh();
};
refresh();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    data_dir: Path
    labels: Path

    def log_message(self, fmt, *args):  # 安静一点
        pass

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _series_dirs(self) -> list[Path]:
        return sorted(d for d in self.data_dir.iterdir() if d.is_dir()) if self.data_dir.exists() else []

    def _done(self) -> dict[str, int]:
        out: dict[str, int] = {}
        if self.labels.exists():
            with self.labels.open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    out[r["series"]] = out.get(r["series"], 0) + 1
        return out

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/series":
            done = self._done()
            self._json([{"name": d.name, "done": d.name in done, "n": done.get(d.name, 0)} for d in self._series_dirs()])
            return

        if path.startswith("/api/series/"):
            name = urllib.parse.unquote(path.split("/", 3)[3])
            d = self.data_dir / name
            if not d.is_dir():
                self._json({"error": "no such series"}, 404)
                return
            files = sorted(p.name for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            self._json({"series": name, "files": files})
            return

        if path.startswith("/img/"):
            parts = path.split("/", 3)
            name, fname = urllib.parse.unquote(parts[2]), urllib.parse.unquote(parts[3])
            f = (self.data_dir / name / fname).resolve()
            if not str(f).startswith(str(self.data_dir.resolve())) or not f.exists():
                self.send_error(404)
                return
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path != "/api/label":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            series = payload["series"]
            items = [(f, int(r)) for f, r in payload["items"]]
            if not series or not items:
                raise ValueError("空数据")
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 400)
            return

        existing: list[dict] = []
        if self.labels.exists():
            with self.labels.open(encoding="utf-8") as fh:
                existing = [r for r in csv.DictReader(fh) if r["series"] != series]
        existing += [{"series": series, "file": f, "rank": r} for f, r in sorted(items, key=lambda t: t[1])]

        self.labels.parent.mkdir(parents=True, exist_ok=True)
        with self.labels.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["series", "file", "rank"])
            w.writeheader()
            w.writerows(existing)
        self._json({"ok": True, "n": len(items), "file": str(self.labels)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/series")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    Handler.data_dir = Path(args.data).resolve()
    Handler.labels = Path(args.labels)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"标注台已启动： http://127.0.0.1:{args.port}   （Ctrl+C 退出）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
