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
import argparse, base64, itertools, json, os, re, subprocess, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = "results"
MOVIES_ROOT = "data/movies"
CANON_RESULTS = "canonical/canonical_results.json"
_SESSION_RE = re.compile(r"^[\w-]+$")
_MAX_REFINE_FRAMES = 400
_THUMB_W = 200
_VIDEO_EXTS = (".mkv", ".mp4", ".mov", ".avi", ".m4v", ".webm")

# background label_prep.py runs, keyed by an incrementing job id
_prep_lock = threading.Lock()
_prep_jobs = {}
_prep_ids = itertools.count(1)

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
<header><a id=backlink href="/" style="color:#7fb0e8;text-decoration:none">&larr; movie</a>
 <b>Cut labeller</b><select id=session></select><span id=name></span><span class=sp></span>
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
  if(M.video) document.getElementById('backlink').href='/?v='+encodeURIComponent(M.video);
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


HOME_PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>Cut library</title>
<style>
 body{background:#141414;color:#ddd;font:14px system-ui,sans-serif;margin:0}
 header{position:sticky;top:0;background:#1d1d1d;padding:12px 16px;border-bottom:1px solid #333;
  display:flex;gap:14px;align-items:center;z-index:10}
 header b{font-size:17px} .sp{flex:1} a{color:#7fb0e8;text-decoration:none} a:hover{text-decoration:underline}
 .wrap{max-width:1080px;margin:0 auto;padding:16px}
 h1{font-size:20px;margin:4px 0 2px} h2{font-size:15px;color:#bbb;border-bottom:1px solid #2a2a2a;padding-bottom:6px;margin:26px 0 12px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}
 .card{border:1px solid #333;border-radius:10px;padding:12px;background:#1b1b1b}
 .card.click{cursor:pointer} .card.click:hover{border-color:#4679b8;background:#1f2530}
 .card .nm{font-weight:600;font-size:14px;margin-bottom:2px}
 .card .sub{color:#888;font-size:12px;margin-bottom:8px}
 svg.diag{width:100%;aspect-ratio:1/1;height:auto;background:#101010;border-radius:6px;display:block}
 .diagref{stroke:#3d4a5c;stroke-width:1;stroke-dasharray:3 4}
 .diaggrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin-top:10px}
 .diagcell .celllab{font-size:10px;color:#7d8590;text-align:center;margin-top:2px}
 .cthumb{height:52px;width:auto;border-radius:3px;background:#000;flex:none}
 .rng{position:relative;flex:1;min-width:240px;height:26px}
 .rng .rtrack{position:absolute;left:0;right:0;top:11px;height:4px;background:#333;border-radius:2px}
 .rng .rfill{position:absolute;top:11px;height:4px;background:#4679b8;border-radius:2px}
 .rng input[type=range]{position:absolute;left:0;top:0;width:100%;height:26px;margin:0;background:none;pointer-events:none;-webkit-appearance:none;appearance:none}
 .rng input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;pointer-events:auto;height:18px;width:18px;border-radius:50%;background:#7fb0e8;border:2px solid #141414;cursor:pointer}
 .rng input[type=range]::-moz-range-thumb{pointer-events:auto;height:16px;width:16px;border-radius:50%;background:#7fb0e8;border:2px solid #141414;cursor:pointer}
 .rng input[type=range]::-webkit-slider-runnable-track{background:none} .rng input[type=range]::-moz-range-track{background:none}
 .gbtn{font-size:12px;padding:5px 8px} .gbtn.on.gacc{background:#1f6f3f;border-color:#2c9;color:#eafff2}
 .gbtn.on.grej{background:#7a2d29;border-color:#c66;color:#ffe6e3}
 .overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:50;align-items:center;justify-content:center;padding:20px}
 .pbox{background:#1b1b1b;border:1px solid #333;border-radius:10px;padding:14px;max-width:900px;width:100%}
 .prow{display:flex;gap:10px;align-items:center;margin-bottom:8px}
 .row{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
 button,select,input{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:6px 9px;font:inherit}
 button{cursor:pointer} button:hover{background:#333} button:disabled{opacity:.5;cursor:default}
 .go{background:#1f6f3f;border-color:#2c9;color:#eafff2} .go:hover{background:#268a4f}
 .badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px}
 .b-done{background:#1f5c37;color:#bfe6cd} .b-prog{background:#6b5320;color:#f0d79a}
 .b-todo{background:#3a3a3a;color:#bbb} .b-none{background:#4a2a2a;color:#e0a}
 .box{border:1px solid #333;border-radius:10px;padding:14px;background:#1b1b1b;margin-bottom:8px}
 .fr{display:flex;gap:10px;align-items:end;flex-wrap:wrap}
 .fr label{display:flex;flex-direction:column;gap:4px;font-size:12px;color:#999}
 .fr input[type=number]{width:100px}
 .joblog{white-space:pre-wrap;font:12px ui-monospace,monospace;color:#9c9;background:#101010;
  border-radius:6px;padding:8px;margin-top:10px;max-height:170px;overflow:auto;display:none}
 .muted{color:#777;font-size:13px}
 .fixrow{display:flex;gap:10px;align-items:center;border:1px solid #2c2c2c;border-radius:8px;padding:8px 10px;margin:6px 0;background:#181818}
 .fixrow .fn{flex:1;font-weight:600} .biglabel{max-width:520px;margin:10px auto 0}
 .tabs{display:inline-flex;border:1px solid #444;border-radius:6px;overflow:hidden}
 .tabs button{border:none;border-radius:0;background:#222;color:#999;font-size:12px;padding:5px 12px}
 .tabs button.on{background:#2f4a6b;color:#dce9f8}
 .clus{border:1px solid #2c2c2c;border-radius:8px;background:#181818;margin:6px 0;padding:8px 10px}
 .clus.settled{opacity:.62}
 .crow{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 .crank{color:#777;min-width:24px;text-align:right;font-size:12px}
 .crow .fn{font-weight:600;font-size:14px}
 .cmem{margin:6px 0 0 34px}
 .cstrip{position:relative;height:14px;flex:0 0 120px;background:#101010;border:1px solid #2a2a2a;border-radius:4px;overflow:hidden}
 .cstrip i{position:absolute;top:2px;bottom:2px;width:2px;border-radius:1px}
 .tl{position:relative;height:46px;background:#101010;border:1px solid #333;border-radius:6px;margin:12px 0;cursor:crosshair;user-select:none;touch-action:none;overflow:hidden}
 .tl .tick{position:absolute;top:8px;bottom:8px;width:1px;background:#3987e5;opacity:.4}
 .tl .sel{position:absolute;top:0;bottom:0;background:rgba(127,176,232,.22);border-left:2px solid #7fb0e8;border-right:2px solid #7fb0e8;pointer-events:none}
 .tl .lab{position:absolute;bottom:3px;font-size:10px;color:#bcd;pointer-events:none;background:rgba(16,16,16,.75);padding:0 3px;border-radius:3px}
</style></head><body>
<header><a href="/" onclick="go(null,event)"><b>Cut library</b></a>
 <span class=muted id=crumb></span><span class=sp></span></header>
<div class=wrap id=app></div>
<script>
const RAMP=['#3987e5','#4e93e8','#5598e7','#6da7ec','#86b6ef','#9ec5f4','#b7d3f6','#cde2fb'];
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function attr(s){return esc(s).replace(/'/g,"&#39;");}
function fmtT(s){s=+s;const m=Math.floor(s/60),ss=s-60*m;return m>0?`${m}m${ss.toFixed(1)}s`:`${ss.toFixed(2)}s`;}
// Diagonal shot blocks: each shot is a square sized by its duration, tiled
// corner-to-corner up the main diagonal. Colour = duration (validated blue
// sequential ramp, dim=short → bright=long) on the dark surface.
// core: render one diagonal from an already-prepared, sorted list of cut boundaries `cc`.
function diagSvg(cc,size){
  const durs=[];for(let i=0;i<cc.length-1;i++)durs.push(cc[i+1]-cc[i]);
  const T=durs.reduce((a,b)=>a+b,0);
  if(!(T>0)||!durs.length) return `<svg viewBox="0 0 ${size} ${size}" class="diag"></svg>`;
  const sc=size/T;
  const logs=durs.map(d=>Math.log(Math.max(d,1e-3)));
  const lo=Math.min(...logs),hi=Math.max(...logs),span=(hi-lo)||1;
  let cum=0,rects='';
  for(let i=0;i<durs.length;i++){
    const d=durs[i], s=Math.max(d*sc,1), x=cum*sc, y=size-(cum+d)*sc;
    const col=RAMP[Math.round(((logs[i]-lo)/span)*(RAMP.length-1))];
    rects+=`<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${s.toFixed(2)}" height="${s.toFixed(2)}" rx="0.6" fill="${col}"><title>@${fmtT(cc[i])} · ${d.toFixed(2)}s</title></rect>`;
    cum+=d;
  }
  return `<svg viewBox="0 0 ${size} ${size}" class="diag">
    <line x1="0" y1="${size}" x2="${size}" y2="0" class="diagref"></line>${rects}</svg>`;
}
// prep: interior cuts sorted, first/last 10 dropped (opening idents / end credits).
function diagCuts(cuts,start,end){
  let cc=(cuts||[]).filter(t=>t>start&&t<end).slice().sort((a,b)=>a-b);
  if(cc.length>25) cc=cc.slice(10,-10);
  return cc;
}
// small single diagonal (movie cards, at-a-glance)
function diag(cuts,start,end,size){ return diagSvg(diagCuts(cuts,start,end), size||240); }
// big view: split the film into square panels (~200 shots each) read left→right, top→bottom
// so each shot gets far more room than one collapsed diagonal.
function diagPanels(cuts,start,end){
  const cc=diagCuts(cuts,start,end);
  const nShots=cc.length-1;
  if(nShots<40) return `<div class=biglabel>${diagSvg(cc,300)}</div>`;
  const panels=Math.max(2,Math.min(20,Math.round(nShots/200)));
  const per=Math.ceil(nShots/panels);
  let cells='';
  for(let p=0;p<panels;p++){
    const a=p*per, seg=cc.slice(a, Math.min(cc.length, a+per+1));  // +1 shares the boundary cut
    if(seg.length<2) continue;
    cells+=`<div class=diagcell>${diagSvg(seg,180)}<div class=celllab>${fmtT(seg[0])}–${fmtT(seg[seg.length-1])}</div></div>`;
  }
  return `<div class=diaggrid>${cells}</div>`;
}
function fixBadge(f){
  return f.labelled?'<span class="badge b-done">labelled</span>'
    :f.in_progress?'<span class="badge b-prog">in progress</span>'
    :'<span class="badge b-todo">not started</span>';
}

// ---- movie list (landing) ----
function movieCard(m){
  const nf = m.fixes.length?` · ${m.fixes.length} fix${m.fixes.length>1?'es':''}`:'';
  const body = m.processed
    ? diag(m.cuts,m.start,m.end,240)+`<div class=sub style="margin-top:6px">${m.n_cuts} cuts · ${fmtT(m.end)}${nf}</div>`
    : `<div class=muted style="padding:20px 0;text-align:center">not run yet — click to process →</div>`;
  const badge = m.processed?'<span class="badge b-done">processed</span>':'<span class="badge b-none">unprocessed</span>';
  return `<div class="card click" onclick="go('${attr(m.video)}')">
    <div class=nm>${esc(m.name)} ${badge}</div><div class=sub>${esc(m.slug)}</div>${body}</div>`;
}
async function renderList(){
  document.getElementById('crumb').textContent='movies';
  const app=document.getElementById('app');
  app.innerHTML='<div class=muted>loading…</div>';
  const j=await (await fetch('/api/movies')).json();
  const proc=j.movies.filter(m=>m.processed), todo=j.movies.filter(m=>!m.processed);
  app.innerHTML=`<h1>Movies</h1>
    <h2>Processed <span class=muted style="font-weight:400">(${proc.length})</span></h2>
    <div class=grid>${proc.map(movieCard).join('')||'<div class=muted>None processed yet.</div>'}</div>
    <h2>Unprocessed <span class=muted style="font-weight:400">(${todo.length})</span></h2>
    <div class=muted>Movies found in your folders that have no base run yet — click one to process it. Add more folders with <code>MOVIE_DIRS=/path/a,/path/b</code> in a <code>.env</code> file next to label_server.py.</div>
    <div class=grid style="margin-top:12px">${todo.map(movieCard).join('')||'<div class=muted>None — every movie found is processed.</div>'}</div>`;
}

// ---- movie detail ----
async function renderMovie(video){
  const app=document.getElementById('app');
  app.innerHTML='<div class=muted>loading…</div>';
  const m=await (await fetch('/api/movie?v='+encodeURIComponent(video))).json();
  if(m.error){ app.innerHTML='<div class=muted>'+esc(m.error)+'</div>'; return; }
  document.getElementById('crumb').innerHTML='<a href="/" onclick="go(null,event)">movies</a> › '+esc(m.name);
  window._movie=m;
  let html=`<div class=row style="margin-bottom:4px"><a href="/" onclick="go(null,event)">← movies</a></div>
    <h1>${esc(m.name)} ${m.processed?'<span class="badge b-done">processed</span>':'<span class="badge b-none">not processed</span>'}</h1>
    <div class=sub>${esc(m.slug)}</div>`;

  if(m.processed){
    window._ncuts=m.n_cuts;
    html+=`<div class=box style="margin-top:12px"><div class=sub>End result — base run (${esc(m.base_source)}) with saved fixes applied · <span id=ncuts>${m.n_cuts}</span> cuts · ${fmtT(m.end)}</div>
      <div class=muted style="margin-top:4px">shot lengths as squares along each diagonal; panels run in order (first/last 10 cuts trimmed)</div>
      ${diagPanels(m.cuts,m.start,m.end)}</div>`;
    html+=`<div class=box style="margin-top:12px">
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:6px;flex-wrap:wrap">
        <div style="font-weight:600">Least-confident cuts</div><span class=sp></span>
        <div class=tabs><button id=tab_list class=on onclick="showTab('list')">List</button><button id=tab_clusters onclick="showTab('clusters')">Clusters</button></div>
      </div>
      <div id=panel_list>
        <div class=muted>Two handles set the confidence window shown — the low handle defaults to the
          accept threshold (0.5). Accepting or rejecting a cut removes it from this list; use
          "Decided" to revisit what you've already accepted/rejected.</div>
        <div id=weakbands class=muted style="margin-top:6px">loading…</div>
        <div class=tabs style="margin-top:8px"><button id=wt_und class=on onclick="weakMode('undecided')">To review</button><button id=wt_dec onclick="weakMode('decided')">Decided</button></div>
        <div id=weakrngrow class=row style="margin-top:10px;align-items:center">
          <span class=muted>conf</span>
          <div class=rng id=weakrng>
            <div class=rtrack></div><div class=rfill id=weakfill></div>
            <input type=range id=weaklo min=0.3 max=0.85 step=0.005 value=0.5 oninput="weakSlide('lo')">
            <input type=range id=weakhi min=0.3 max=0.85 step=0.005 value=0.85 oninput="weakSlide('hi')">
          </div>
          <span id=weaklabel class=muted style="min-width:190px"></span>
        </div>
        <div id=weaklist style="margin-top:8px;max-height:340px;overflow:auto"></div>
      </div>
      <div id=panel_clusters style="display:none">
        <div class=muted>Bursts of weak cuts packed close in time — usually a flashing/strobing scene shedding
          dozens of false cuts at once. Biggest cleanups first: one click accepts or rejects a whole burst;
          expand a cluster to override single cuts.</div>
        <div class=row style="margin-top:6px">
          <label class=muted>conf below <select id=cconf onchange="loadClusters()">
            <option>0.4</option><option>0.5</option><option selected>0.6</option><option>0.7</option><option>0.85</option></select></label>
          <label class=muted>max gap <select id=cgap onchange="loadClusters()">
            <option>0.75</option><option>1</option><option selected>1.5</option><option>2.5</option><option>4</option></select> s</label>
          <span id=cinfo class=muted></span>
        </div>
        <div id=clusterlist style="margin-top:8px;max-height:520px;overflow:auto"></div>
      </div>
    </div>`;
  } else {
    html+=`<div class=box style="margin-top:12px"><div class=sub>This movie has no base run yet. Processing runs whole-film detection (TransNetV2) — this can take a while for a full film.</div>
      <div class=row><button class=go id=procbtn onclick="processMovie()">Process movie</button></div>
      <div class=joblog id=procLog></div></div>`;
  }

  const durTxt = m.duration ? fmtTC(m.duration) : 'unknown';
  html+=`<h2>Scene fixes</h2>
    <div class=muted>A fix hand-labels the cuts in one scene; saved fixes override the base run there (base + fixes = the end result above).</div>
    <div id=fixlist style="margin-top:10px"></div>
    <div class=box style="margin-top:12px">
      <div style="font-weight:600;margin-bottom:6px">Add a scene fix</div>
      <div class=muted>Drag across the timeline to pick the scene — or type timecodes (m:ss or h:mm:ss). Movie length: ${durTxt}.</div>
      <div class=tl id=tl></div>
      <div class=fr>
        <label>name<input id=fname type=text placeholder="e.g. tower_chase"></label>
        <label>start<input id=fstart type=text placeholder="0:00" oninput="syncFromFields()"></label>
        <label>end<input id=fend type=text placeholder="1:30" oninput="syncFromFields()"></label>
        <button class=go id=fbtn onclick="addFix()">Detect &amp; label scene</button>
      </div>
      <div class=joblog id=fixLog></div>
    </div>`;
  app.innerHTML=html;
  renderFixes(m);
  setupTimeline(m);
  if(m.processed){ loadWeak(); loadClusters(); if(location.hash==='#clusters') showTab('clusters'); }
}
function showTab(which){
  document.getElementById('panel_list').style.display = which==='list'?'':'none';
  document.getElementById('panel_clusters').style.display = which==='clusters'?'':'none';
  document.getElementById('tab_list').classList.toggle('on', which==='list');
  document.getElementById('tab_clusters').classList.toggle('on', which==='clusters');
  history.replaceState(null,'','/?v='+encodeURIComponent(window._movie.video)+(which==='clusters'?'#clusters':''));
}

let _weakMode='undecided';
async function loadWeak(){
  const m=window._movie;
  const bands=document.getElementById('weakbands');
  const j=await (await fetch('/api/weak?v='+encodeURIComponent(m.video))).json();
  if(!j.total){ bands.textContent='No per-cut confidence for this base run (only detector runs carry it, not imported canonicals).'; return; }
  window._weakall=j.rows||[]; const b=j.bands;
  bands.innerHTML=`of ${j.total} cuts — <b style="color:#e0a">&lt;0.4:</b> ${b.lt04} · <b style="color:#f0d79a">0.4–0.6:</b> ${b.b0406} · <b>0.6–0.8:</b> ${b.b0608} · <b style="color:#bfe6cd">≥0.8:</b> ${b.gte08} (solid)`;
  weakSlide();
}
function weakMode(mode){
  _weakMode=mode;
  document.getElementById('wt_und').classList.toggle('on', mode==='undecided');
  document.getElementById('wt_dec').classList.toggle('on', mode==='decided');
  document.getElementById('weakrngrow').style.display = mode==='undecided'?'':'none';
  weakSlide();
}
function cutRow(x){ const v=encodeURIComponent(window._movie.video); return `<div class=fixrow>
    <img class=cthumb loading=lazy src="/frameat?v=${v}&t=${(x.time-0.15).toFixed(3)}" title="frame before">
    <img class=cthumb loading=lazy src="/frameat?v=${v}&t=${(x.time+0.15).toFixed(3)}" title="frame after">
    <div class=fn>${fmtTC(x.time)}</div>
    <div class=sub style="margin:0;flex:1"><span class="badge ${x.kind==='gradual'?'b-prog':'b-todo'}">${x.kind}</span> conf ${x.conf}</div>
    <button onclick="watchCut(${x.time})">▶ watch</button>
    <button class="gbtn gacc ${x.gold==='accept'?'on':''}" onclick="setGold(${x.time},'accept')" title="gold: real cut">✓ accept</button>
    <button class="gbtn grej ${x.gold==='reject'?'on':''}" onclick="setGold(${x.time},'reject')" title="gold: not a cut">✗ reject</button>
    <button onclick="reviewCut(${x.time})">fix →</button></div>`;
}
function weakSlide(which){
  const loEl=document.getElementById('weaklo'), hiEl=document.getElementById('weakhi');
  if(!loEl||!hiEl) return;
  const all=window._weakall||[];
  const nU=all.filter(r=>!r.gold).length, nD=all.filter(r=>r.gold).length;
  const bu=document.getElementById('wt_und'), bd=document.getElementById('wt_dec');
  if(bu) bu.textContent='To review ('+nU+')'; if(bd) bd.textContent='Decided ('+nD+')';
  let sub;
  if(_weakMode==='decided'){
    sub=all.filter(r=>r.gold).slice().sort((a,b)=>a.conf-b.conf);
    document.getElementById('weaklabel').textContent=`${sub.length} decided`;
  } else {
    let lo=+loEl.value, hi=+hiEl.value;
    if(lo>hi){ if(which==='hi') loEl.value=(lo=hi); else hiEl.value=(hi=lo); }  // handles can't cross
    const MIN=0.3, MAX=0.85, fill=document.getElementById('weakfill');
    if(fill){ fill.style.left=((lo-MIN)/(MAX-MIN)*100)+'%'; fill.style.right=((MAX-hi)/(MAX-MIN)*100)+'%'; }
    sub=all.filter(r=>!r.gold && r.conf>=lo && r.conf<=hi);
    document.getElementById('weaklabel').textContent=`conf ${lo.toFixed(3)}–${hi.toFixed(3)} · ${sub.length} to review`;
  }
  document.getElementById('weaklist').innerHTML=sub.map(cutRow).join('');
}
function bumpNcuts(d){   // arithmetic live update; server recomputes exactly on reload
  if(!d) return; window._ncuts=(window._ncuts||0)+d;
  const el=document.getElementById('ncuts'); if(el) el.textContent=window._ncuts;
}
function findClusterCut(t){
  for(const c of (window._clusters||[])){ const x=c.members.find(y=>y.time===t); if(x) return x; }
  return null;
}
async function setGold(t, decision){
  const m=window._movie, row=(window._weakall||[]).find(r=>r.time===t), cm=findClusterCut(t);
  const cur=row?row.gold:(cm?cm.gold:null);
  const dec=(cur===decision) ? 'clear' : decision;   // click again to un-set
  await fetch('/api/gold',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({video:m.video, time:t, decision:dec})});
  const g = dec==='clear' ? null : dec;
  if(row) row.gold=g; if(cm) cm.gold=g;   // a verdict moves the cut out of "To review" into "Decided"
  bumpNcuts((cur==='reject'?1:0)-(g==='reject'?1:0));   // rejects drop cuts from the end result
  weakSlide(); renderClusterList();
}
function watchCut(t){ openClip(t, 2, 0.08, '@'+fmtTC(t), 'plays the 2s up to the cut and ends on the cut frame', [t], true); }
function openClip(t, pre, post, label, note, marks, onlyMark){
  const m=window._movie;
  const q='v='+encodeURIComponent(m.video)+'&t='+t+'&pre='+pre.toFixed(2)+'&post='+post.toFixed(2);
  const vq=q+((marks&&marks.length)?'&mark='+marks.map(x=>x.toFixed(3)).join(','):'')+(onlyMark?'&only=1':'');
  let ov=document.getElementById('player');
  if(!ov){ ov=document.createElement('div'); ov.id='player'; ov.className='overlay';
    ov.addEventListener('click',e=>{ if(e.target===ov) closePlayer(); }); document.body.appendChild(ov); }
  ov.innerHTML=`<div class=pbox onclick="event.stopPropagation()">
    <div class=prow><b>${label}</b><span class=muted>${note}</span><span class=sp></span><button onclick="closePlayer()">close ✕</button></div>
    <video controls autoplay muted playsinline style="width:100%;max-height:70vh;background:#000">
      <source src="/clip?${q}" type="video/mp4"><track default kind=subtitles srclang=en src="/clipvtt?${vq}"></video>
    <div class=muted style="margin-top:6px">Transcoding a ${Math.round(pre+post)}s clip. Muted autoplay — unmute in the controls.</div></div>`;
  ov.style.display='flex';
}

// ---- clusters view ----
async function loadClusters(){
  const m=window._movie, el=document.getElementById('clusterlist');
  if(!el) return;
  const conf=document.getElementById('cconf').value, gap=document.getElementById('cgap').value;
  el.innerHTML='<div class=muted>loading…</div>';
  const j=await (await fetch('/api/clusters?v='+encodeURIComponent(m.video)+'&conf='+conf+'&gap='+gap)).json();
  window._clusters=j.clusters||[]; window._copen=new Set();
  document.getElementById('tab_clusters').textContent='Clusters ('+window._clusters.length+')';
  const info=document.getElementById('cinfo');
  if(!j.total_low && !j.clusters.length){
    info.textContent='';
    el.innerHTML='<div class=muted>No low-confidence cuts to cluster (imported canonicals carry no per-cut confidence).</div>';
    return;
  }
  info.textContent=`${j.n_clustered} of ${j.total_low} weak cuts fall in ${j.clusters.length} cluster${j.clusters.length===1?'':'s'} (${j.n_single} singles → List view)`;
  renderClusterList();
}
function renderClusterList(){
  const cs=window._clusters||[], el=document.getElementById('clusterlist');
  if(!el) return;
  el.innerHTML = cs.length ? cs.map(clusterCard).join('')
    : '<div class=muted>No bursts of ≥3 weak cuts within the gap at these settings — loosen conf/gap, or review singles in the List view.</div>';
}
function clusterCard(c,i){
  const nR=c.members.filter(x=>x.gold==='reject').length, nA=c.members.filter(x=>x.gold==='accept').length;
  const allR=nR===c.count, allA=nA===c.count, open=window._copen.has(i);
  const state = allR ? '<span class="badge b-none">✗ all rejected</span>'
    : allA ? '<span class="badge b-done">✓ all accepted</span>'
    : (nR||nA) ? `<span class="badge b-prog">${nA}✓ · ${nR}✗ of ${c.count}</span>` : '';
  const strip=c.members.map(x=>{
    const L=c.span>0 ? ((x.time-c.start)/c.span*94+2) : 50;
    const col=x.gold==='reject' ? '#c66' : x.gold==='accept' ? '#3c9' : '#d99c3f';
    return `<i style="left:${L.toFixed(1)}%;background:${col}"></i>`;
  }).join('');
  return `<div class="clus ${(allR||allA)?'settled':''}">
    <div class=crow>
      <div class=crank>#${i+1}</div>
      <div style="min-width:150px"><div class=fn>${c.count} cuts in ${c.span.toFixed(1)}s</div>
        <div class=sub style="margin:0">${fmtTC(c.start)}–${fmtTC(c.end)} · conf ${c.conf_min.toFixed(2)}–${c.conf_max.toFixed(2)}</div></div>
      <div class=cstrip title="cut positions across the burst (amber=undecided, red=rejected, green=accepted)">${strip}</div>
      ${state}<span class=sp></span>
      <button onclick="watchCluster(${i})">▶ watch burst</button>
      <button class="gbtn gacc ${allA?'on':''}" onclick="goldCluster(${i},'accept')" title="gold: every cut in this burst is real${allA?' (click again to clear)':''}">✓ accept all</button>
      <button class="gbtn grej ${allR?'on':''}" onclick="goldCluster(${i},'reject')" title="gold: every cut in this burst is false${allR?' (click again to clear)':''}">✗ reject all</button>
      <button onclick="togCluster(${i})">${open?'▾ hide':'▸ cuts'}</button>
    </div>
    ${open?`<div class=cmem>${c.members.map(cutRow).join('')}</div>`:''}
  </div>`;
}
function togCluster(i){ window._copen.has(i)?window._copen.delete(i):window._copen.add(i); renderClusterList(); }
async function goldCluster(i, decision){
  const m=window._movie, c=window._clusters[i];
  const dec=c.members.every(x=>x.gold===decision) ? 'clear' : decision;   // re-click undoes
  await fetch('/api/gold_batch',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({video:m.video, times:c.members.map(x=>x.time), decision:dec})});
  const before=c.members.filter(x=>x.gold==='reject').length;
  const g = dec==='clear' ? null : dec;
  c.members.forEach(x=>{ x.gold=g; const r=(window._weakall||[]).find(y=>y.time===x.time); if(r) r.gold=g; });
  bumpNcuts(before - c.members.filter(x=>x.gold==='reject').length);
  renderClusterList(); weakSlide();
}
function watchCluster(i){
  const c=window._clusters[i], mid=(c.start+c.end)/2, pad=2.5;
  openClip(mid, Math.min(30, mid-c.start+pad), Math.min(60, c.end-mid+pad),
    `${c.count}-cut burst · ${fmtTC(c.start)}–${fmtTC(c.end)}`, 'each ◆ THIS CUT marks a burst cut being judged', c.members.map(x=>x.time));
}
function closePlayer(){ const ov=document.getElementById('player'); if(ov){ ov.style.display='none'; ov.innerHTML=''; } }
function reviewCut(t){
  window._sel={start:Math.max(0,t-8), end:t+8};
  if(document.getElementById('tlsel')) paintSel();
  writeFields();
  const nm=document.getElementById('fname');
  if(nm){ if(!nm.value) nm.value='cut_'+Math.round(t)+'s'; nm.scrollIntoView({behavior:'smooth',block:'center'}); nm.focus(); }
}
// timecode helpers: "1:23:45" / "83:45" / "12.5" -> seconds, and back
function parseTC(s){
  s=String(s).trim(); if(!s) return NaN;
  if(s.indexOf(':')<0) return parseFloat(s);
  const p=s.split(':'); if(p.some(x=>x===''||isNaN(+x))) return NaN;
  return p.reduce((a,x)=>a*60+(+x),0);
}
function fmtTC(t){
  t=Math.max(0,Math.round(t)); const h=Math.floor(t/3600),m=Math.floor(t%3600/60),s=t%60;
  return (h?h+':':'')+String(m).padStart(h?2:1,'0')+':'+String(s).padStart(2,'0');
}
function setupTimeline(m){
  const tl=document.getElementById('tl'); if(!tl) return;
  const D=window._dur=m.duration||m.end||0;
  if(!D){ tl.style.display='none';
    document.getElementById('fstart').value='0:00'; document.getElementById('fend').value='1:30';
    window._sel={start:0,end:90}; return; }
  let ticks='';
  for(const c of (m.cuts||[])) if(c>0&&c<=D) ticks+=`<div class=tick style="left:${(c/D*100).toFixed(3)}%"></div>`;
  tl.innerHTML=ticks+'<div class=sel id=tlsel></div><div class=lab id=tlL style="left:2px"></div><div class=lab id=tlR style="right:2px"></div>';
  window._sel={start:0,end:Math.min(90,D)};
  const frac=e=>{const r=tl.getBoundingClientRect();return Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));};
  let dragging=false,anchor=0;
  tl.addEventListener('pointerdown',e=>{dragging=true;anchor=frac(e)*D;window._sel={start:anchor,end:anchor};tl.setPointerCapture(e.pointerId);paintSel();});
  tl.addEventListener('pointermove',e=>{if(!dragging)return;const t=frac(e)*D;window._sel={start:Math.min(anchor,t),end:Math.max(anchor,t)};paintSel();writeFields();});
  tl.addEventListener('pointerup',()=>{dragging=false;if(window._sel.end-window._sel.start<0.5)window._sel.end=Math.min(D,window._sel.start+60);paintSel();writeFields();});
  writeFields(); paintSel();
}
function paintSel(){
  const D=window._dur,s=window._sel,sel=document.getElementById('tlsel'); if(!sel||!D)return;
  sel.style.left=(s.start/D*100)+'%'; sel.style.width=Math.max(0.4,(s.end-s.start)/D*100)+'%';
  document.getElementById('tlL').textContent=fmtTC(s.start);
  document.getElementById('tlR').textContent=fmtTC(s.end);
}
function writeFields(){
  document.getElementById('fstart').value=fmtTC(window._sel.start);
  document.getElementById('fend').value=fmtTC(window._sel.end);
}
function syncFromFields(){
  const s=parseTC(document.getElementById('fstart').value),e=parseTC(document.getElementById('fend').value);
  if(!isNaN(s)&&!isNaN(e)){ window._sel={start:s,end:e}; paintSel(); }
}
function renderFixes(m){
  const el=document.getElementById('fixlist');
  if(!m.fixes.length){ el.innerHTML='<div class=muted>No scene fixes yet.</div>'; return; }
  el.innerHTML=m.fixes.map(f=>`<div class=fixrow>
    <div class=fn>${esc(f.name)}</div>
    <div class=sub style="margin:0">${fmtT(f.start)}–${fmtT(f.end)} · ${f.n_cuts} cuts ${fixBadge(f)}</div>
    <a href="/label?s=${encodeURIComponent(f.name)}"><button>open in labeller →</button></a>
  </div>`).join('');
}

// ---- jobs ----
function etxt(s){s=s||0;const m=Math.floor(s/60);return (m?m+'m':'')+(s%60)+'s';}
function jobPoll(job, logEl, btn, onDone){
  fetch('/api/prep_status?job='+job).then(r=>r.json()).then(j=>{
    const body=(j.tail||'').trim();
    if(j.status==='running'){
      logEl.textContent=(body?body+'\n':'')+'⏳ working… '+etxt(j.elapsed)+' elapsed (a full film takes a few minutes — leave this open)';
      logEl.scrollTop=logEl.scrollHeight;
      setTimeout(()=>jobPoll(job,logEl,btn,onDone),1500); return;
    }
    if(btn){ btn.disabled=false; btn.textContent=btn.dataset.label; }
    logEl.textContent=body||('('+j.status+')');
    if(j.status==='done') onDone(j); else logEl.textContent+='\n\n✗ failed (exit '+j.returncode+') after '+etxt(j.elapsed);
  });
}
function processMovie(){
  const m=window._movie, btn=document.getElementById('procbtn'), log=document.getElementById('procLog');
  btn.dataset.label='Process movie'; btn.disabled=true; btn.textContent='processing…';
  log.style.display='block'; log.textContent='starting whole-film detection — this takes a few minutes for a full film…';
  fetch('/api/process',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video:m.video})})
    .then(r=>r.json().then(j=>({ok:r.ok,j}))).then(({ok,j})=>{
      if(!ok){ log.textContent='error: '+(j.error||'failed'); btn.disabled=false; btn.textContent='Process movie'; return; }
      jobPoll(j.job,log,btn,()=>{ log.textContent+='\n\n✓ base run ready. Reloading…'; setTimeout(()=>renderMovie(m.video),900); });
    });
}
function addFix(){
  const m=window._movie;
  const name=document.getElementById('fname').value.trim();
  const start=parseTC(document.getElementById('fstart').value);
  const end=parseTC(document.getElementById('fend').value);
  const log=document.getElementById('fixLog'), btn=document.getElementById('fbtn');
  log.style.display='block';
  if(!name||!/^[\w-]+$/.test(name)){ log.textContent='name must be letters/digits/_/- only'; return; }
  if(isNaN(start)||isNaN(end)||!(end>start)){ log.textContent='enter valid start/end timecodes (end after start)'; return; }
  btn.dataset.label='Detect & label scene'; btn.disabled=true; btn.textContent='detecting…';
  log.textContent='detecting candidate cuts…';
  fetch('/api/prep',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({video:m.video,start,end,name})})
    .then(r=>r.json().then(j=>({ok:r.ok,j}))).then(({ok,j})=>{
      if(!ok){ log.textContent='error: '+(j.error||'failed'); btn.disabled=false; btn.textContent='Detect & label scene'; return; }
      jobPoll(j.job,log,btn,(jj)=>{ log.textContent+='\n\n✓ ready — opening labeller…'; setTimeout(()=>{location.href='/label?s='+encodeURIComponent(jj.name);},900); });
    });
}

// ---- routing ----
function go(video, ev){
  if(ev) ev.preventDefault();
  const url = video?('/?v='+encodeURIComponent(video)):'/';
  history.pushState({video},'',url);
  route();
}
function route(){
  const v=new URLSearchParams(location.search).get('v');
  if(v) renderMovie(v); else renderList();
}
window.onpopstate=route;
route();
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


def _session_names():
    out = []
    if os.path.isdir(ROOT):
        for entry in sorted(os.listdir(ROOT)):
            if entry.startswith("label_") and os.path.isfile(os.path.join(ROOT, entry, "candidates.json")):
                out.append(entry[len("label_"):])
    return out


def _session_cuts(name):
    """Current cut list for a session's diagonal viz: canonical.json if SAVEd, else
    the working.json / candidates.json 'kept' set the labeller would show."""
    d = _session_dir(name)
    if not d:
        return None
    meta = json.load(open(os.path.join(d, "candidates.json")))
    cands = {c["id"]: c for c in meta["candidates"]}
    canon_fp, work_fp = os.path.join(d, "canonical.json"), os.path.join(d, "working.json")
    labelled, in_progress = os.path.isfile(canon_fp), os.path.isfile(work_fp)
    if labelled:
        times = sorted(json.load(open(canon_fp)).get("cuts", []))
    else:
        if in_progress:
            work = json.load(open(work_fp))
            for c in work.get("customCandidates", []):
                cands[c["id"]] = c
            kept_ids = set(work.get("kept", []))
        else:
            kept_ids = {c["id"] for c in meta["candidates"] if c.get("kept")}
        times = sorted(cands[i]["time"] for i in kept_ids if i in cands)
    return {"name": name, "video": meta.get("video"), "start": meta["start"],
            "end": meta["end"], "fps": meta.get("fps"), "cuts": times,
            "n_candidates": len(meta["candidates"]), "n_cuts": len(times),
            "labelled": labelled, "in_progress": in_progress}


_MOVIE_DIRS = None


def _load_env(path=".env"):
    """Minimal .env reader (no dependency): KEY=VALUE lines, # comments, optional quotes."""
    env = {}
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _movie_dirs():
    """Folders to scan for movies: data/movies plus any MOVIE_DIRS in .env (comma or
    newline separated). Cached for the process lifetime -- restart to pick up .env edits."""
    global _MOVIE_DIRS
    if _MOVIE_DIRS is None:
        dirs, seen = [], set()
        raw = _load_env().get("MOVIE_DIRS") or _load_env().get("MOVIES_DIR") or ""
        for d in [MOVIES_ROOT, *re.split(r"[,\n]", raw)]:
            d = d.strip()
            if d and os.path.isdir(d):
                rp = os.path.realpath(d)
                if rp not in seen:
                    seen.add(rp); dirs.append(d)
        _MOVIE_DIRS = dirs
    return _MOVIE_DIRS


def _movie_files():
    """Video files across all configured movie folders, de-duplicated by real path."""
    out, seen = [], set()
    for root in _movie_dirs():
        for dp, _, fns in os.walk(root):
            for fn in sorted(fns):
                if fn.lower().endswith(_VIDEO_EXTS):
                    v = os.path.join(dp, fn)
                    rp = os.path.realpath(v)
                    if rp not in seen:
                        seen.add(rp); out.append(v)
    return sorted(out)


def _movie_root_of(video):
    """The configured scan folder a video lives under, or None."""
    rp = os.path.realpath(video)
    for root in _movie_dirs():
        base = os.path.realpath(root)
        if rp.startswith(base + os.sep):
            return base
    return None


def _movie_slug(video):
    """Stable id for a movie: its top folder under its scan root, else the filename stem."""
    base = _movie_root_of(video)
    if base:
        name = os.path.realpath(video)[len(base) + 1:].split(os.sep)[0]
    else:
        name = os.path.splitext(os.path.basename(video))[0]
    return re.sub(r"\W+", "_", name).strip("_").lower() or "movie"


def _pretty_name(video):
    base = _movie_root_of(video)
    folder = os.path.realpath(video)[len(base) + 1:].split(os.sep)[0] if base else \
        os.path.splitext(os.path.basename(video))[0]
    return folder.replace(".", " ").strip()


def _canonical_by_path():
    if not os.path.isfile(CANON_RESULTS):
        return {}
    out = {}
    for e in json.load(open(CANON_RESULTS)).values():
        if e.get("video"):
            out[os.path.realpath(e["video"])] = e
    return out


def _base_cuts(video):
    """The whole-film base run for a movie: canonical_results if known, else cut_times'
    own <stem>.cuts.json next to the video (adopted directly, so a movie shows processed
    the moment detection finishes -- no dependence on the web layer's polling), else a
    results/base/<slug>.json copy. Returns (cuts, source|None)."""
    e = _canonical_by_path().get(os.path.realpath(video))
    if e is not None:
        return sorted(e.get("cuts") or []), "canonical"
    src = video.rsplit(".", 1)[0] + ".cuts.json"
    if os.path.isfile(src):
        try:
            doc = json.load(open(src))
            cuts = doc.get("cuts") or []
            if cuts and (doc.get("analysis") or {}).get("status") != "failed":
                return sorted(cuts), "run"
        except (OSError, ValueError):
            pass
    bf = os.path.join("results", "base", _movie_slug(video) + ".json")
    if os.path.isfile(bf):
        return sorted(json.load(open(bf)).get("cuts") or []), "run"
    return [], None


def _base_events(video):
    """The base run's per-cut events (with confidence), from cut_times' <stem>.cuts.json.
    Empty for imported canonicals, which carry only bare times."""
    src = video.rsplit(".", 1)[0] + ".cuts.json"
    if os.path.isfile(src):
        try:
            return json.load(open(src)).get("cut_events") or []
        except (OSError, ValueError):
            pass
    return []


def _weak_cuts(video, ceiling=0.85):
    """The base run's cuts ranked weakest-first, plus confidence-band counts. `review` is
    the tail below `ceiling` (cuts above that are solid and not worth reviewing) -- the
    slider in the UI walks down this list."""
    rows = []
    for e in _base_events(video):
        c = e.get("confidence") or {}
        v = c.get("value")
        if v is None:
            continue
        rows.append({"time": e["time"], "conf": round(v, 3), "kind": e.get("transition_kind")})
    rows.sort(key=lambda r: r["conf"])
    st = _gold_state(video)
    for r in rows:
        r["gold"] = st.get(round(r["time"], 3))
    # only cuts still ACCEPTED are worth reviewing: those in the end result (base + fixes)
    # minus gold rejects. A rejected cut is already handled, so it drops off the list.
    base, src = _base_cuts(video)
    accepted = set()
    if src:
        rej = {round(t, 3) for t in _load_gold(video)["reject"]}
        accepted = {round(t, 3) for t in _end_result(base, _movie_fixes(video))
                    if round(t, 3) not in rej}
    bands = {"lt04": sum(1 for r in rows if r["conf"] < 0.4),
             "b0406": sum(1 for r in rows if 0.4 <= r["conf"] < 0.6),
             "b0608": sum(1 for r in rows if 0.6 <= r["conf"] < 0.8),
             "gte08": sum(1 for r in rows if r["conf"] >= 0.8)}
    # reviewable weak cuts: those still in the end result (accepted, undecided) OR already
    # carrying a verdict (so the "Decided" view can show rejects, which left the end result).
    review = [r for r in rows if r["conf"] < ceiling and (round(r["time"], 3) in accepted or r["gold"])]
    return {"total": len(rows), "ceiling": ceiling, "bands": bands, "rows": review}


def _cut_clusters(video, conf_max=0.6, gap=1.5, min_size=3):
    """Low-confidence cuts grouped into time-dense bursts. Flash/strobe scenes (and shaky
    gradual regions) shed runs of weak detections packed close together -- the same
    pathologically-dense regions postfilter.dampen_strobe targets. A cluster is a maximal
    run of cuts with conf < conf_max where consecutive members are <= gap seconds apart
    and the run has >= min_size members; shorter runs stay singletons for the List view.
    Ranked by impact: most members first (biggest one-click cleanup), then weakest mean
    confidence, then earliest."""
    low = []
    for e in _base_events(video):
        v = (e.get("confidence") or {}).get("value")
        if v is not None and v < conf_max:
            low.append({"time": e["time"], "conf": round(v, 3),
                        "kind": e.get("transition_kind")})
    low.sort(key=lambda r: r["time"])
    st = _gold_state(video)
    for r in low:
        r["gold"] = st.get(round(r["time"], 3))
    runs, run = [], []
    for r in low:
        if run and r["time"] - run[-1]["time"] > gap:
            runs.append(run); run = []
        run.append(r)
    if run:
        runs.append(run)
    clusters = []
    for mem in runs:
        if len(mem) < min_size:
            continue
        confs = [m["conf"] for m in mem]
        clusters.append({"start": mem[0]["time"], "end": mem[-1]["time"],
                         "span": round(mem[-1]["time"] - mem[0]["time"], 3),
                         "count": len(mem), "conf_min": min(confs), "conf_max": max(confs),
                         "conf_mean": round(sum(confs) / len(confs), 3), "members": mem})
    clusters.sort(key=lambda c: (-c["count"], c["conf_mean"], c["start"]))
    n_in = sum(c["count"] for c in clusters)
    return {"total_low": len(low), "conf_max": conf_max, "gap": gap, "min_size": min_size,
            "n_clustered": n_in, "n_single": len(low) - n_in, "clusters": clusters}


GOLD_ROOT = os.path.join("results", "gold")


def _gold_path(video):
    return os.path.join(GOLD_ROOT, _movie_slug(video) + ".json")


def _load_gold(video):
    """Per-movie human verdicts on individual base-run cuts: {accept:[t], reject:[t]}."""
    fp = _gold_path(video)
    if os.path.isfile(fp):
        try:
            g = json.load(open(fp))
            return {"accept": list(g.get("accept") or []), "reject": list(g.get("reject") or [])}
        except (OSError, ValueError):
            pass
    return {"accept": [], "reject": []}


def _set_gold_batch(video, times, decision):
    """Record accept / reject / clear for many cut times in one read+write of the gold
    file (a time carries at most one verdict). Same file format as single verdicts."""
    g = _load_gold(video)
    keys = {round(float(t), 3) for t in times}
    g["accept"] = [t for t in g["accept"] if round(t, 3) not in keys]
    g["reject"] = [t for t in g["reject"] if round(t, 3) not in keys]
    if decision in ("accept", "reject"):
        g[decision].extend(float(t) for t in times)
    os.makedirs(GOLD_ROOT, exist_ok=True)
    json.dump(g, open(_gold_path(video), "w"), indent=1)
    return g


def _set_gold(video, time, decision):
    """Record accept / reject / clear for one cut time."""
    return _set_gold_batch(video, [time], decision)


def _gold_state(video):
    g = _load_gold(video)
    st = {round(t, 3): "accept" for t in g["accept"]}
    st.update({round(t, 3): "reject" for t in g["reject"]})
    return st


def _vtt_ts(s):
    s = max(0.0, s)
    h = int(s // 3600); m = int(s % 3600 // 60)
    return f"{h:02d}:{m:02d}:{s % 60:06.3f}"


def _clip_vtt(video, t, pre, post, marks=None, only_marks=False):
    """WebVTT for a preview clip [t-pre, t+post], in clip-relative time.

    only_marks=True (the single-cut watch): emit ONE caption, at the cut under review, and
    nothing else -- so the only subtitle you see is the cut in question. Otherwise: one cue
    per shot labelled with the running scene index (bumps at every cut), with any `marks`
    (the cut/s being judged) flagged so you know which transition matters in a busy window."""
    cuts = sorted(_base_cuts(video)[0])
    cs, ce = max(0.0, t - pre), t + post
    if only_marks:
        # single-cut watch: caption the WHOLE clip so "◆ THIS CUT" is visible on every frame
        # (including the cut frame at the end), regardless of when the track finishes loading.
        if any(cs <= m <= ce for m in (marks or [])):
            return "WEBVTT\n\n" + f"{_vtt_ts(0)} --> {_vtt_ts(ce - cs + 0.5)}\n◆ THIS CUT\n"
        return "WEBVTT\n"
    mk = {round(x, 2) for x in (marks or [])}
    bounds = [cs] + [c for c in cuts if cs < c < ce] + [ce]
    out = ["WEBVTT", ""]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        shot = sum(1 for c in cuts if c <= (a + b) / 2) + 1
        label = f"◆ THIS CUT → Shot {shot}" if round(a, 2) in mk else f"Shot {shot}"
        out += [f"{_vtt_ts(a - cs)} --> {_vtt_ts(b - cs)}", label, ""]
    return "\n".join(out)


def _movie_fixes(video):
    """Labelling sessions whose source video is this movie -- the scene fixes for it."""
    rp = os.path.realpath(video)
    out = [sc for sc in map(_session_cuts, _session_names())
           if sc and sc["video"] and os.path.realpath(sc["video"]) == rp]
    out.sort(key=lambda s: s["start"])
    return out


def _end_result(base_cuts, fixes):
    """base + fixes = end result: each SAVED (labelled) fix replaces the base cuts inside
    its [start, end] region with its own hand-verified cuts. Unsaved fixes don't count."""
    cuts = list(base_cuts)
    for f in fixes:
        if not f["labelled"]:
            continue
        s, e = f["start"], f["end"]
        cuts = [c for c in cuts if not (s <= c <= e)] + [c for c in f["cuts"] if s <= c <= e]
    return sorted({round(c, 4) for c in cuts})


def _probe_duration(video):
    """Whole-film length in seconds via ffprobe, or None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", video],
            capture_output=True, text=True).stdout.strip()
        d = float(out)
        return d if d > 0 else None
    except (OSError, ValueError):
        return None


def _movie_detail(video, probe=False):
    base, source = _base_cuts(video)
    fixes = _movie_fixes(video)
    end = _end_result(base, fixes) if source else []
    gold = _load_gold(video)
    rej = {round(t, 3) for t in gold["reject"]}
    if rej:                                # gold rejects drop false cuts from the end result
        end = [t for t in end if round(t, 3) not in rej]
    span_end = max(end) if end else (max(base) if base else 0.0)
    canon = _canonical_by_path().get(os.path.realpath(video))
    name = (canon.get("name") if canon else None) or _pretty_name(video)
    duration = _probe_duration(video) if probe else None
    return {"video": video, "name": name, "slug": _movie_slug(video),
            "processed": source is not None, "base_source": source,
            "base_cuts": base, "cuts": end, "start": 0.0, "end": span_end,
            "duration": duration or span_end, "n_cuts": len(end), "fixes": fixes,
            "gold": {"accept": len(gold["accept"]), "reject": len(gold["reject"])}}


def _all_movies():
    """Every movie under data/movies plus any canonical entry, most-relevant first."""
    seen, movies = set(), []
    for e in _canonical_by_path().values():
        if e.get("video") and os.path.isfile(e["video"]):
            seen.add(os.path.realpath(e["video"])); movies.append(e["video"])
    for v in _movie_files():
        if os.path.realpath(v) not in seen:
            seen.add(os.path.realpath(v)); movies.append(v)
    out = [_movie_detail(v) for v in movies]
    out.sort(key=lambda m: (not m["processed"], m["name"].lower()))
    return out


def _launch(cmd, name, kind, **extra):
    logf = tempfile.NamedTemporaryFile(prefix=f"{kind}_{name}_", suffix=".log", delete=False)
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    logf.close()
    with _prep_lock:
        jid = next(_prep_ids)
        _prep_jobs[jid] = {"proc": proc, "name": name, "log": logf.name,
                           "kind": kind, "final": False, "t0": time.time(), **extra}
    return jid


def _running_process_for(video):
    """The job id of an in-flight base run for this movie, or None (avoids double runs)."""
    rp = os.path.realpath(video)
    with _prep_lock:
        for jid, job in _prep_jobs.items():
            if (job.get("kind") == "process" and job.get("video")
                    and os.path.realpath(job["video"]) == rp and job["proc"].poll() is None):
                return jid
    return None


def _valid_video(video):
    if not (video and os.path.isfile(video)):
        return "video not found"
    rp = os.path.realpath(video)
    roots = [os.path.realpath("data")] + [os.path.realpath(d) for d in _movie_dirs()]
    if any(rp.startswith(r + os.sep) for r in roots):
        return None
    return "video is outside the configured movie folders"


def _start_prep(video, start, end, name):
    """Launch label_prep.py over [start, end] to create a scene fix; returns (job, error)."""
    if not (name and _SESSION_RE.match(name)):
        return None, "invalid fix name"
    err = _valid_video(video)
    if err:
        return None, err
    cmd = ["python3", "label_prep.py", "--video", video, "--name", name,
           "--start", str(start), "--end", str(end)]
    return _launch(cmd, name, "prep"), None


def _start_process(video):
    """Launch a whole-film base run (cut_times.py transnet); returns (job, error).

    Decode defaults to transnet-CPU: TransNetV2 inference runs on the GPU either way, and
    for typical 720p/1080p sources CPU decode is ~2x faster than the NVDEC+hwdownload path
    (which leaves the GPU mostly idle). Override with PROCESS_DETECTOR in .env, e.g. set it
    to transnet-cuda for 4K/10-bit HEVC where NVDEC pays off."""
    err = _valid_video(video)
    if err:
        return None, err
    existing = _running_process_for(video)      # don't start a second run for the same movie
    if existing:
        return existing, None
    slug = _movie_slug(video)
    detector = _load_env().get("PROCESS_DETECTOR") or "transnet-cpu"
    cmd = ["python3", "cut_times.py", video, "--detector", detector, "--format", "json"]
    return _launch(cmd, slug, "process", video=video, slug=slug), None


def _finalize_process(job):
    """After a base run finishes, validate cut_times' output and copy it into
    results/base/. Returns (ok, detail): a decode that failed or found no cuts does NOT
    become a base run -- we surface it as an error instead of a silent empty 'processed'."""
    stem = job["video"].rsplit(".", 1)[0]
    src = stem + ".cuts.json"
    if not os.path.isfile(src):
        return False, "detector wrote no output"
    doc = json.load(open(src))
    cuts = doc.get("cuts") or []
    status = (doc.get("analysis") or {}).get("status")
    if status == "failed" or not cuts:
        warns = (doc.get("analysis") or {}).get("warnings") or []
        why = warns[0] if warns else "detection produced 0 cuts"
        return False, f"detection failed ({why.splitlines()[0][:160]})"
    os.makedirs(os.path.join("results", "base"), exist_ok=True)
    json.dump({"video": job["video"], "cuts": cuts,
               "fps": doc.get("fps"), "detector": doc.get("detector")},
              open(os.path.join("results", "base", job["slug"] + ".json"), "w"), indent=1)
    return True, None


def _prep_status(jid):
    with _prep_lock:
        job = _prep_jobs.get(jid)
    if not job:
        return None
    rc = job["proc"].poll()
    status = "running" if rc is None else ("done" if rc == 0 else "error")
    if status == "done" and job.get("kind") == "process" and not job.get("final"):
        try:
            ok, detail = _finalize_process(job)
        except (OSError, ValueError, KeyError) as e:
            ok, detail = False, str(e)
        job["final"] = True
        if not ok:                       # ran but produced nothing usable -> not "done"
            job["error"] = detail
    try:
        with open(job["log"], "rb") as fh:
            tail = fh.read()[-1600:].decode("utf-8", "replace")
    except OSError:
        tail = ""
    if job.get("error"):
        status = "error"
        tail = (tail + "\n\n" + job["error"]).strip()
    return {"job": jid, "name": job["name"], "kind": job.get("kind"),
            "status": status, "returncode": rc, "tail": tail,
            "elapsed": int(time.time() - job.get("t0", time.time()))}


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
            return self._send(200, "text/html; charset=utf-8", HOME_PAGE.encode())
        if p == "/label":
            return self._send(200, "text/html; charset=utf-8", PAGE.encode())

        if p == "/api/movies":
            return self._json(200, {"movies": _all_movies()})
        if p == "/api/movie":
            v = qs.get("v", [None])[0]
            if not (v and os.path.isfile(v)):
                return self._json(404, {"error": "unknown movie"})
            return self._json(200, _movie_detail(v, probe=True))
        if p == "/api/weak":
            v = qs.get("v", [None])[0]
            if not (v and os.path.isfile(v)):
                return self._json(404, {"error": "unknown movie"})
            return self._json(200, _weak_cuts(v))
        if p == "/api/clusters":
            v = qs.get("v", [None])[0]
            if not (v and os.path.isfile(v)):
                return self._json(404, {"error": "unknown movie"})
            try:
                conf = float(qs.get("conf", ["0.6"])[0])
                gap = float(qs.get("gap", ["1.5"])[0])
                msz = max(2, int(qs.get("min", ["3"])[0]))
            except ValueError:
                return self._json(400, {"error": "bad params"})
            return self._json(200, _cut_clusters(v, conf, gap, msz))
        if p == "/frameat":
            v = qs.get("v", [None])[0]
            if not (v and os.path.isfile(v)):
                return self._send(404, "text/plain", b"unknown movie")
            try:
                t = max(0.0, float(qs["t"][0]))
            except (KeyError, ValueError):
                return self._send(400, "text/plain", b"bad t")
            frames = _decode_frames(v, t, 1, width=150)
            if not frames:
                return self._send(404, "text/plain", b"decode failed")
            return self._send(200, "image/jpeg", base64.b64decode(frames[0]))

        if p == "/clipvtt":
            v = qs.get("v", [None])[0]
            if not (v and os.path.isfile(v)):
                return self._send(404, "text/plain", b"unknown movie")
            try:
                t = float(qs["t"][0]); pre = float(qs.get("pre", ["5"])[0]); post = float(qs.get("post", ["8"])[0])
            except (KeyError, ValueError):
                return self._send(400, "text/plain", b"bad params")
            marks = []
            for s in qs.get("mark", [""])[0].split(","):
                try:
                    marks.append(float(s))
                except ValueError:
                    pass
            only = qs.get("only", ["0"])[0] == "1"
            return self._send(200, "text/vtt", _clip_vtt(v, t, pre, post, marks, only).encode())

        if p == "/clip":
            v = qs.get("v", [None])[0]
            if not (v and os.path.isfile(v)):
                return self._send(404, "text/plain", b"unknown movie")
            try:
                t = float(qs["t"][0])
                pre = min(30.0, max(0.0, float(qs.get("pre", ["5"])[0])))
                post = min(60.0, max(0.0, float(qs.get("post", ["8"])[0])))
            except (KeyError, ValueError):
                return self._send(400, "text/plain", b"bad params")
            # transcode a short window to browser-safe H.264 mp4 (works for any source
            # container/codec, incl. mkv/hevc), streamed straight to the client. Two-stage
            # seek: a fast keyframe seek before -i, then an ACCURATE fine seek after -i, so
            # the window is exactly [t-pre, t-pre+dur] -- otherwise a keyframe-rounded start
            # drifts the clip (and the subtitle timed to it) off by up to a GOP.
            start = max(0.0, t - pre)
            coarse = max(0.0, start - 5.0)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                   "-ss", str(coarse), "-i", v, "-ss", str(start - coarse), "-t", str(pre + post),
                   "-vf", "scale='min(854,iw)':-2",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-c:a", "aac",
                   "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.end_headers()
            try:
                for chunk in iter(lambda: proc.stdout.read(65536), b""):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                proc.kill()
            finally:
                proc.stdout.close(); proc.wait()
            return

        if p == "/api/prep_status":
            try:
                jid = int(qs["job"][0])
            except (KeyError, ValueError):
                return self._json(400, {"error": "bad job id"})
            st = _prep_status(jid)
            return self._json(200, st) if st else self._json(404, {"error": "unknown job"})

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

        if p == "/api/prep":
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n)) if n else {}
            try:
                start, end = float(data["start"]), float(data["end"])
            except (KeyError, ValueError, TypeError):
                return self._json(400, {"error": "bad start/end"})
            if not (end > start):
                return self._json(400, {"error": "end must be greater than start"})
            jid, err = _start_prep(data.get("video"), start, end, data.get("name"))
            if err:
                return self._json(400, {"error": err})
            return self._json(200, {"job": jid})

        if p == "/api/process":
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n)) if n else {}
            jid, err = _start_process(data.get("video"))
            if err:
                return self._json(400, {"error": err})
            return self._json(200, {"job": jid})

        if p == "/api/gold":
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n)) if n else {}
            v = data.get("video")
            if not (v and os.path.isfile(v)):
                return self._json(404, {"error": "unknown movie"})
            try:
                t = float(data["time"])
            except (KeyError, ValueError, TypeError):
                return self._json(400, {"error": "bad time"})
            g = _set_gold(v, t, data.get("decision"))
            return self._json(200, {"accept": len(g["accept"]), "reject": len(g["reject"])})

        if p == "/api/gold_batch":
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n)) if n else {}
            v = data.get("video")
            if not (v and os.path.isfile(v)):
                return self._json(404, {"error": "unknown movie"})
            try:
                times = [float(t) for t in data["times"]]
            except (KeyError, ValueError, TypeError):
                return self._json(400, {"error": "bad times"})
            if not times:
                return self._json(400, {"error": "empty times"})
            g = _set_gold_batch(v, times, data.get("decision"))
            return self._json(200, {"accept": len(g["accept"]), "reject": len(g["reject"])})

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
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    srv.daemon_threads = True          # requests handled concurrently; don't block the UI
    srv.serve_forever()
