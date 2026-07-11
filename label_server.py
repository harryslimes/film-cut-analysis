"""Local web tool to record canonical cut labels for a sequence.

  python label_server.py --dir results/label_vertigo_nightmare --port 8000
  then open http://localhost:8000

- Kept cuts show before/after frames; click REJECT to mark a wrong keep.
- Each gap has a '+ expand' button revealing lower-confidence candidates (ranked by
  probability) so you can PROMOTE a missed cut.
- SAVE writes canonical.json (+ canonical.srt) into the --dir.
"""
import argparse, json, os
from http.server import BaseHTTPRequestHandler, HTTPServer

DIR = "."

PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>Cut labeller</title>
<style>
 body{background:#141414;color:#ddd;font:14px system-ui,sans-serif;margin:0}
 header{position:sticky;top:0;background:#1d1d1d;padding:10px 16px;border-bottom:1px solid #333;
  display:flex;gap:16px;align-items:center;z-index:10}
 header b{font-size:16px} .sp{flex:1}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:6px 10px;cursor:pointer}
 button:hover{background:#333}
 #save{background:#1f6f3f;border-color:#2c9} #save:hover{background:#268a4f}
 .wrap{max-width:900px;margin:0 auto;padding:16px}
 .card{display:flex;gap:10px;align-items:center;border:2px solid #2c8c4a;border-radius:8px;
  padding:8px;margin:8px 0;background:#1b1b1b}
 .card.cand{border-color:#c98b2a;opacity:.95}
 .card img{height:96px;border-radius:4px;background:#000}
 .meta{flex:1} .meta .t{font-size:15px;font-weight:600} .meta .p{color:#999;font-size:12px}
 .gap{margin:4px 0;text-align:center}
 .gap button{background:#222;border-style:dashed;color:#aaa;font-size:12px}
 .rej{background:#5a2320;border-color:#a44}
 .add{background:#234b2b;border-color:#4a8}
 .badge{display:inline-block;padding:1px 6px;border-radius:10px;font-size:11px;margin-left:6px}
 .bs{background:#2a3a5a} .bg{background:#3a2a5a}
</style></head><body>
<header><b>Cut labeller</b><span id=name></span><span class=sp></span>
 <span id=stat></span><button id=save>SAVE canonical</button></header>
<div class=wrap id=app></div>
<script>
let M=null, kept=new Set(), expanded=new Set();
async function load(){ M=await (await fetch('candidates.json')).json();
  document.getElementById('name').textContent=M.name+'  ('+M.start+'-'+M.end+'s)';
  M.candidates.forEach(c=>{ if(c.kept) kept.add(c.id); }); render(); }
function thumb(c){ return `<img src="thumbs/${c.id}_b.jpg" title="before"><img src="thumbs/${c.id}_a.jpg" title="after">`; }
function keptCard(c){ return `<div class=card><div style="display:flex">${thumb(c)}</div>
  <div class=meta><div class=t>${c.time.toFixed(2)}s</div>
   <div class=p>conf ${c.prob} <span class="badge bs">sharp ${c.sharp}</span><span class="badge bg">grad ${c.gradual}</span></div></div>
  <button class=rej onclick="rej(${c.id})">✗ reject</button></div>`; }
function candCard(c){ return `<div class="card cand"><div style="display:flex">${thumb(c)}</div>
  <div class=meta><div class=t>${c.time.toFixed(2)}s</div>
   <div class=p>conf ${c.prob} <span class="badge bs">sharp ${c.sharp}</span><span class="badge bg">grad ${c.gradual}</span></div></div>
  <button class=add onclick="add(${c.id})">+ keep</button></div>`; }
function render(){
  const cs=M.candidates.slice().sort((a,b)=>a.time-b.time);
  const keptList=cs.filter(c=>kept.has(c.id));
  let html=''; let prevT=M.start;
  const gaps=[]; // boundaries: [start, keptcut, keptcut, ..., end]
  const bounds=[M.start, ...keptList.map(c=>c.time), M.end];
  for(let g=0; g<bounds.length-1; g++){
    const lo=bounds[g], hi=bounds[g+1];
    // candidates shown in TIME order (their real position in the gap); conf ranks the summary
    const inGap=cs.filter(c=>!kept.has(c.id) && c.time>lo && c.time<hi).sort((a,b)=>a.time-b.time);
    const gid='g'+g;
    if(inGap.length){
      const topConf=Math.max(...inGap.map(c=>c.prob));
      html+=`<div class=gap><button onclick="tog('${gid}')">${expanded.has(gid)?'▾ hide':'+ expand'} ${inGap.length} candidate${inGap.length>1?'s':''} (top conf ${topConf})</button></div>`;
      if(expanded.has(gid)) inGap.forEach(c=> html+=candCard(c));
    }
    if(g<keptList.length) html+=keptCard(keptList[g]);
  }
  document.getElementById('app').innerHTML=html;
  document.getElementById('stat').textContent=kept.size+' kept / '+cs.length+' candidates';
}
function rej(id){kept.delete(id);render();}
function add(id){kept.add(id);render();}
function tog(g){expanded.has(g)?expanded.delete(g):expanded.add(g);render();}
document.getElementById('save').onclick=async()=>{
  const times=M.candidates.filter(c=>kept.has(c.id)).map(c=>c.time).sort((a,b)=>a-b);
  const r=await fetch('save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({kept:[...kept],times})});
  const j=await r.json(); document.getElementById('save').textContent='saved ✓ '+j.n+' cuts';
  setTimeout(()=>document.getElementById('save').textContent='SAVE canonical',1500);
};
load();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, ctype, body):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/" or p == "/index.html":
            return self._send(200, "text/html; charset=utf-8", PAGE.encode())
        if p == "/candidates.json":
            return self._send(200, "application/json", open(os.path.join(DIR, "candidates.json"), "rb").read())
        if p.startswith("/thumbs/"):
            fp = os.path.join(DIR, p.lstrip("/"))
            if os.path.exists(fp):
                return self._send(200, "image/jpeg", open(fp, "rb").read())
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path == "/save":
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            meta = json.load(open(os.path.join(DIR, "candidates.json")))
            times = sorted(data["times"])
            json.dump({"video": meta["video"], "start": meta["start"], "end": meta["end"],
                       "fps": meta["fps"], "cuts": times},
                      open(os.path.join(DIR, "canonical.json"), "w"), indent=1)
            # write a canonical SRT for the region
            def ts(s):
                ms = int(round(s*1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); sec, ms = divmod(ms, 1000)
                return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
            b = [meta["start"]] + times + [meta["end"]]
            with open(os.path.join(DIR, "canonical.srt"), "w") as f:
                for i in range(len(b)-1):
                    f.write(f"{i+1}\n{ts(b[i])} --> {ts(b[i+1])}\nShot {i+1}\n\n")
            return self._send(200, "application/json", json.dumps({"n": len(times)}).encode())
        self._send(404, "text/plain", b"not found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(); DIR = a.dir
    print(f"serving {DIR} at http://localhost:{a.port}  (Ctrl-C to stop)")
    HTTPServer(("0.0.0.0", a.port), H).serve_forever()
