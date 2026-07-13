"""Local web tool to record canonical cut labels for a sequence.

  python label_server.py --root results --port 8000
  then open http://localhost:8000 and pick a session from the dropdown

--root points at the parent directory holding one or more label_<name>/ session
folders (each produced by label_prep.py). The page lists them in a dropdown so you
don't need to relaunch the server to switch movies/scenes.

- Kept cuts show before/after frames; click REJECT to mark a wrong keep.
- Each gap has a '+ expand' button revealing lower-confidence candidates (ranked by
  probability) so you can PROMOTE a missed cut.
- Wherever two adjacent shown cuts (kept or expanded candidates) aren't frame-adjacent,
  a '+ refine' button appears with the exact frame count still unresolved between them.
  It opens a frame-accurate slider (plus increment buttons) over just those frames --
  decoded fresh from the source video in one batch (candidates.json only ships
  thumbnails for the peaks TransNetV2 already flagged, not every frame in between).
  Very wide unresolved spans fall back to on-demand single-frame decoding instead of
  batch-decoding hundreds of frames.
- Every keep/reject/refine-commit is autosaved to working.json so a forgotten SAVE or a
  closed tab doesn't lose progress; reopening the page resumes from it. DISCARD wipes
  it and resets to the original candidates.json state.
- SAVE writes canonical.json (+ canonical.srt) into the session's folder.
"""
import argparse, base64, json, os, re, subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = "results"
_SESSION_RE = re.compile(r"^[\w-]+$")
_MAX_REFINE_FRAMES = 400
_THUMB_W = 200

PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>Cut labeller</title>
<style>
 body{background:#141414;color:#ddd;font:14px system-ui,sans-serif;margin:0}
 header{position:sticky;top:0;background:#1d1d1d;padding:10px 16px;border-bottom:1px solid #333;
  display:flex;gap:16px;align-items:center;z-index:10}
 header b{font-size:16px} .sp{flex:1}
 select{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:6px 10px}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:6px 10px;cursor:pointer}
 button:hover{background:#333}
 #save{background:#1f6f3f;border-color:#2c9} #save:hover{background:#268a4f}
 #discard{background:#5a2320;border-color:#a44} #discard:hover{background:#742d29}
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
 .customrow{margin:4px 0;text-align:center}
 .customrow button{background:#222;border-style:dashed;color:#aaa;font-size:12px}
 .custom input[type=range]{width:180px;vertical-align:middle}
</style></head><body>
<header><b>Cut labeller</b><select id=session></select><span id=name></span><span class=sp></span>
 <span id=stat></span><button id=discard onclick="discardAll()">discard &amp; restart</button>
 <button id=save>SAVE canonical</button></header>
<div class=wrap id=app></div>
<script>
let M=null, SESSION=null, kept=new Set(), expanded=new Set();
let refineOpen=new Set(), refineVal={}, refineFrames={}, nextCustomId=1000000;

async function initSessions(){
  const j = await (await fetch('sessions')).json();
  const sel = document.getElementById('session');
  sel.innerHTML = j.sessions.map(s=>`<option value="${s}">${s}</option>`).join('');
  sel.onchange = ()=>switchSession(sel.value);
  if(!j.sessions.length){
    document.getElementById('app').innerHTML =
      `<p style="padding:20px;color:#999">No label_* sessions found under ${j.root}. Run label_prep.py first.</p>`;
    return;
  }
  const urlS = new URLSearchParams(location.search).get('s');
  SESSION = (urlS && j.sessions.includes(urlS)) ? urlS : j.sessions[0];
  sel.value = SESSION;
  await load();
}
function switchSession(name){
  SESSION=name; kept=new Set(); expanded=new Set();
  refineOpen=new Set(); refineVal={}; refineFrames={}; nextCustomId=1000000;
  load();
}
async function load(){
  history.replaceState(null,'','?s='+encodeURIComponent(SESSION));
  M=await (await fetch('candidates.json?s='+encodeURIComponent(SESSION))).json();
  document.getElementById('name').textContent='('+M.start+'-'+M.end+'s)';
  const wr=await fetch('working.json?s='+encodeURIComponent(SESSION));
  const working=wr.ok?await wr.json():null;
  if(working){
    (working.customCandidates||[]).forEach(c=>M.candidates.push(c));
    kept=new Set(working.kept||[]);
    nextCustomId=Math.max(nextCustomId, working.nextCustomId||0);
  } else {
    M.candidates.forEach(c=>{ if(c.kept) kept.add(c.id); });
  }
  render();
}
function autosave(){
  const customCandidates=M.candidates.filter(c=>c.manual);
  fetch('autosave?s='+encodeURIComponent(SESSION),{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({kept:[...kept], customCandidates, nextCustomId})});
}
async function discardAll(){
  if(!confirm('Discard all progress and restart from the original candidates?')) return;
  await fetch('discard?s='+encodeURIComponent(SESSION),{method:'POST'});
  kept=new Set(); expanded=new Set(); refineOpen=new Set(); refineVal={}; refineFrames={};
  M.candidates=M.candidates.filter(c=>!c.manual);
  M.candidates.forEach(c=>{ if(c.kept) kept.add(c.id); });
  render();
}
function thumb(c){
  const s=encodeURIComponent(SESSION);
  if(c.manual) return `<img src="frame?s=${s}&t=${(c.time-0.2).toFixed(3)}" title="before"><img src="frame?s=${s}&t=${(c.time+0.2).toFixed(3)}" title="after">`;
  return `<img src="thumbs/${s}/${c.id}_b.jpg" title="before"><img src="thumbs/${s}/${c.id}_a.jpg" title="after">`;
}
function confMeta(c){
  if(c.manual) return `<div class=p>manual cut</div>`;
  return `<div class=p>conf ${c.prob} <span class="badge bs">sharp ${c.sharp}</span><span class="badge bg">grad ${c.gradual}</span></div>`;
}
function keptCard(c){ return `<div class=card><div style="display:flex">${thumb(c)}</div>
  <div class=meta><div class=t>${c.time.toFixed(3)}s</div>${confMeta(c)}</div>
  <button class=rej onclick="rej(${c.id})">✗ reject</button></div>`; }
function candCard(c){ return `<div class="card cand"><div style="display:flex">${thumb(c)}</div>
  <div class=meta><div class=t>${c.time.toFixed(3)}s</div>${confMeta(c)}</div>
  <button class=add onclick="add(${c.id})">+ keep</button></div>`; }

function refineBetween(a, b){
  const delta=0.2, loT=a.time+delta, hiT=b.time-delta;
  const steps=Math.round((hiT-loT)*M.fps);
  if(steps<1) return '';
  const key='r'+a.id+'_'+b.id;
  if(!refineOpen.has(key)){
    return `<div class=customrow><button onclick="openRefine('${key}',${loT},${steps})">+ refine (${steps} frame${steps>1?'s':''} unresolved)</button></div>`;
  }
  const v=refineVal[key] ?? Math.floor(steps/2);
  const cache=refineFrames[key];
  let imgs, evt;
  if(Array.isArray(cache)){
    imgs=`<img src="data:image/jpeg;base64,${cache[v]}"><img src="data:image/jpeg;base64,${cache[v+1]}">`;
    evt='oninput';
  } else if(cache==='toolarge'){
    const s=encodeURIComponent(SESSION), t=loT+v/M.fps;
    imgs=`<img src="frame?s=${s}&t=${t.toFixed(3)}"><img src="frame?s=${s}&t=${(t+1/M.fps).toFixed(3)}">`;
    evt='onchange';
  } else if(cache==='error'){
    return `<div class="card cand custom"><div class=meta>couldn't decode preview frames
      <button onclick="retryRefine('${key}',${loT},${steps})">retry</button></div></div>`;
  } else {
    return `<div class="card cand custom"><div class=meta>decoding ${steps+1} frames…</div></div>`;
  }
  const t=loT+v/M.fps;
  return `<div class="card cand custom"><div style="display:flex">${imgs}</div>
    <div class=meta><div class=t>${t.toFixed(3)}s &nbsp;frame ${v+1}/${steps}</div>
     <div class=p><button onclick="nudgeRefine('${key}',-1,${steps})">« frame</button>
      <input type=range min=0 max=${steps-1} step=1 value=${v} ${evt}="setRefineVal('${key}',this.value,${steps})">
      <button onclick="nudgeRefine('${key}',1,${steps})">frame »</button></div></div>
    <button class=add onclick="commitRefine('${key}',${loT})">+ keep here</button></div>`;
}
async function openRefine(key, loT, steps){
  refineOpen.add(key);
  if(!(key in refineVal)) refineVal[key]=Math.floor(steps/2);
  render();
  await fetchRefineFrames(key, loT, steps);
}
async function fetchRefineFrames(key, loT, steps){
  try{
    const r=await fetch(`refine_frames?s=${encodeURIComponent(SESSION)}&lo=${loT}&steps=${steps}`);
    if(r.status===413) refineFrames[key]='toolarge';
    else if(r.ok) refineFrames[key]=(await r.json()).frames;
    else refineFrames[key]='error';
  }catch(e){ refineFrames[key]='error'; }
  render();
}
function retryRefine(key, loT, steps){ delete refineFrames[key]; render(); fetchRefineFrames(key, loT, steps); }
function setRefineVal(key, v, steps){ refineVal[key]=Math.max(0, Math.min(steps-1, parseInt(v)||0)); render(); }
function nudgeRefine(key, d, steps){ setRefineVal(key, (refineVal[key] ?? Math.floor(steps/2))+d, steps); }
function commitRefine(key, loT){
  const v=refineVal[key] ?? 0; const t=loT+v/M.fps; const id=nextCustomId++;
  M.candidates.push({id, time:t, manual:true});
  kept.add(id); refineOpen.delete(key); delete refineFrames[key]; delete refineVal[key];
  expanded.clear(); render(); autosave();
}

function render(){
  const cs=M.candidates.slice().sort((a,b)=>a.time-b.time);
  const keptList=cs.filter(c=>kept.has(c.id));
  const bounds=[M.start, ...keptList.map(c=>c.time), M.end];
  let html=''; let lastAnchor=null;
  for(let g=0; g<bounds.length-1; g++){
    const lo=bounds[g], hi=bounds[g+1];
    const inGap=cs.filter(c=>!kept.has(c.id) && c.time>lo && c.time<hi).sort((a,b)=>a.time-b.time);
    const gid='g'+g;
    if(inGap.length){
      const topConf=Math.max(...inGap.map(c=>c.prob));
      html+=`<div class=gap><button onclick="tog('${gid}')">${expanded.has(gid)?'▾ hide':'+ expand'} ${inGap.length} candidate${inGap.length>1?'s':''} (top conf ${topConf})</button></div>`;
      if(expanded.has(gid)){
        inGap.forEach(c=>{
          if(lastAnchor) html+=refineBetween(lastAnchor,c);
          html+=candCard(c); lastAnchor=c;
        });
      }
    }
    if(g<keptList.length){
      const kc=keptList[g];
      if(lastAnchor) html+=refineBetween(lastAnchor,kc);
      html+=keptCard(kc); lastAnchor=kc;
    }
  }
  document.getElementById('app').innerHTML=html;
  document.getElementById('stat').textContent=kept.size+' kept / '+cs.length+' candidates';
}
function rej(id){kept.delete(id);expanded.clear();render();autosave();}
function add(id){kept.add(id);expanded.clear();render();autosave();}
function tog(g){expanded.has(g)?expanded.delete(g):expanded.add(g);render();}
document.getElementById('save').onclick=async()=>{
  const times=M.candidates.filter(c=>kept.has(c.id)).map(c=>c.time).sort((a,b)=>a-b);
  const r=await fetch('save?s='+encodeURIComponent(SESSION),{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({kept:[...kept],times})});
  const j=await r.json(); document.getElementById('save').textContent='saved ✓ '+j.n+' cuts';
  setTimeout(()=>document.getElementById('save').textContent='SAVE canonical',1500);
};
initSessions();
</script></body></html>"""


def _session_dir(name):
    if not name or not _SESSION_RE.match(name):
        return None
    d = os.path.join(ROOT, f"label_{name}")
    return d if os.path.isdir(d) else None


def _decode_frames(video, start_t, n, width=_THUMB_W):
    """Decode n consecutive frames starting at start_t in ONE ffmpeg call (fast+accurate
    two-stage seek), splitting the concatenated MJPEG stream on JPEG SOI/EOI markers.
    Avoids spawning ffmpeg per-frame when scrubbing a whole unresolved cut window."""
    coarse = max(0.0, start_t - 5.0)
    fine = start_t - coarse
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(coarse), "-i", video, "-ss", str(fine),
        "-frames:v", str(n), "-vf", f"scale={width}:-2",
        "-vcodec", "mjpeg", "-f", "mjpeg", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    frames, i = [], 0
    while True:
        start = out.find(b"\xff\xd8", i)
        if start == -1:
            break
        end = out.find(b"\xff\xd9", start)
        if end == -1:
            break
        end += 2
        frames.append(base64.b64encode(out[start:end]).decode("ascii"))
        i = end
    return frames


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, ctype, body):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def _json(self, code, obj):
        self._send(code, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        p = self.path.split("?")[0]
        qs = parse_qs(urlparse(self.path).query)
        sess = qs.get("s", [None])[0]

        if p == "/" or p == "/index.html":
            return self._send(200, "text/html; charset=utf-8", PAGE.encode())

        if p == "/sessions":
            found = []
            if os.path.isdir(ROOT):
                for entry in sorted(os.listdir(ROOT)):
                    full = os.path.join(ROOT, entry)
                    if entry.startswith("label_") and os.path.isfile(os.path.join(full, "candidates.json")):
                        found.append(entry[len("label_"):])
            return self._json(200, {"sessions": found, "root": ROOT})

        d = _session_dir(sess)

        if p == "/candidates.json":
            if not d: return self._send(404, "text/plain", b"unknown session")
            return self._send(200, "application/json", open(os.path.join(d, "candidates.json"), "rb").read())

        if p == "/working.json":
            fp = d and os.path.join(d, "working.json")
            if fp and os.path.exists(fp):
                return self._send(200, "application/json", open(fp, "rb").read())
            return self._send(404, "application/json", b"{}")

        if p.startswith("/thumbs/"):
            rest = p[len("/thumbs/"):]
            if "/" in rest:
                sname, fname = rest.split("/", 1)
                td = _session_dir(sname)
                if td:
                    fp = os.path.join(td, "thumbs", fname)
                    if os.path.exists(fp):
                        return self._send(200, "image/jpeg", open(fp, "rb").read())
            return self._send(404, "text/plain", b"not found")

        if p == "/frame":
            if not d: return self._send(404, "text/plain", b"unknown session")
            try:
                t = max(0.0, float(qs["t"][0]))
            except (KeyError, ValueError):
                return self._send(400, "text/plain", b"bad t")
            meta = json.load(open(os.path.join(d, "candidates.json")))
            frames = _decode_frames(meta["video"], t, 1)
            if not frames:
                return self._send(404, "text/plain", b"decode failed")
            return self._send(200, "image/jpeg", base64.b64decode(frames[0]))

        if p == "/refine_frames":
            if not d: return self._json(404, {"error": "unknown session"})
            try:
                lo = float(qs["lo"][0]); steps = int(qs["steps"][0])
            except (KeyError, ValueError):
                return self._json(400, {"error": "bad params"})
            n = steps + 1
            if n > _MAX_REFINE_FRAMES:
                return self._json(413, {"error": "range too large", "n": n})
            meta = json.load(open(os.path.join(d, "candidates.json")))
            return self._json(200, {"frames": _decode_frames(meta["video"], lo, n)})

        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        p = self.path.split("?")[0]
        qs = parse_qs(urlparse(self.path).query)
        d = _session_dir(qs.get("s", [None])[0])

        if p == "/save":
            if not d: return self._json(404, {"error": "unknown session"})
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            meta = json.load(open(os.path.join(d, "candidates.json")))
            times = sorted(data["times"])
            json.dump({"video": meta["video"], "start": meta["start"], "end": meta["end"],
                       "fps": meta["fps"], "cuts": times},
                      open(os.path.join(d, "canonical.json"), "w"), indent=1)
            def ts(s):
                ms = int(round(s*1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); sec, ms = divmod(ms, 1000)
                return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
            b = [meta["start"]] + times + [meta["end"]]
            with open(os.path.join(d, "canonical.srt"), "w") as f:
                for i in range(len(b)-1):
                    f.write(f"{i+1}\n{ts(b[i])} --> {ts(b[i+1])}\nShot {i+1}\n\n")
            return self._json(200, {"n": len(times)})

        if p == "/autosave":
            if not d: return self._json(404, {"error": "unknown session"})
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            json.dump(data, open(os.path.join(d, "working.json"), "w"), indent=1)
            return self._json(200, {"ok": True})

        if p == "/discard":
            if not d: return self._json(404, {"error": "unknown session"})
            fp = os.path.join(d, "working.json")
            if os.path.exists(fp):
                os.remove(fp)
            return self._json(200, {"ok": True})

        self._send(404, "text/plain", b"not found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results", help="parent dir containing label_<name>/ session folders")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(); ROOT = a.root
    print(f"serving sessions under {ROOT} at http://localhost:{a.port}  (Ctrl-C to stop)")
    HTTPServer(("0.0.0.0", a.port), H).serve_forever()
