"""Score our auto-methods against a saved canonical labelling, and sweep for the best rule.

Loads the candidate pool + your canonical.json, computes median-window CLAHE-SSIM for each
candidate, then: (1) shows how your KEPT cuts differ from REJECTED ones in each feature,
(2) scores existing approaches, (3) sweeps prob/SSIM/density to find the closest rule.
"""
import json, subprocess, argparse
import numpy as np, cv2
from postfilter import dampen_strobe
from evaluate import match

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/label_vertigo_nightmare")
ap.add_argument("--tol", type=float, default=0.3)
a = ap.parse_args()

meta = json.load(open(f"{a.dir}/candidates.json"))
canon = set(round(t, 3) for t in json.load(open(f"{a.dir}/canonical.json"))["cuts"])
V = meta["video"]; A, B, fps = meta["start"], meta["end"], meta["fps"]
cands = meta["candidates"]

# median-window CLAHE-SSIM per candidate
a0 = A - 1.0
raw = subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-ss",str(a0),"-t",str(B-a0+1),
    "-i",V,"-vf","scale=256:144,format=gray","-f","rawvideo","-"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
G = np.frombuffer(raw, np.uint8).reshape([-1,144,256]); clahe = cv2.createCLAHE(2.0,(8,8))
Gc = np.stack([clahe.apply(f) for f in G]).astype(np.float32); n=len(Gc); d=int(round(0.2*fps)); Wn=4
def med(i0,i1): i0=max(0,i0);i1=min(n,max(i1,i0+1));return np.median(Gc[i0:i1],axis=0)
def ssim(x,y):
    x=x.astype(np.float32);y=y.astype(np.float32)
    mx=cv2.GaussianBlur(x,(7,7),1.5);my=cv2.GaussianBlur(y,(7,7),1.5)
    vx=cv2.GaussianBlur(x*x,(7,7),1.5)-mx**2;vy=cv2.GaussianBlur(y*y,(7,7),1.5)-my**2
    vxy=cv2.GaussianBlur(x*y,(7,7),1.5)-mx*my;c1,c2=(2.55)**2,(7.65)**2
    return float((((2*mx*my+c1)*(2*vxy+c2))/((mx**2+my**2+c1)*(vx+vy+c2))).mean())
for c in cands:
    ig=int(round((c["time"]-a0)*fps)); c["ssim"]=round(ssim(med(ig-d-Wn,ig-d),med(ig+d,ig+d+Wn)),3)
    c["gt"]= round(c["time"],3) in canon or any(abs(c["time"]-t)<=0.05 for t in canon)

kept=[c for c in cands if c["gt"]]; rej=[c for c in cands if not c["gt"]]
print(f"canonical: {len(canon)} cuts   candidate pool: {len(cands)}   (tol ±{a.tol}s)\n")
def stats(g,f): v=[c[f] for c in g]; return f"{np.mean(v):.2f}[{np.min(v):.2f}-{np.max(v):.2f}]"
for f in ("prob","sharp","gradual","ssim"):
    print(f"  {f:8} kept={stats(kept,f):22} rejected={stats(rej,f)}")

def score(cuts):
    m=match(sorted(cuts), sorted(canon), tol=a.tol); return m["precision"],m["recall"],m["f1"]

print("\n=== existing approaches vs your canonical ===")
allc=json.load(open(V.rsplit('.',1)[0]+'.cuts.json'))["cuts"]
region_raw=[c for c in allc if A<=c<=B]
print(f"  raw TransNet (region)      : P={score(region_raw)[0]:.2f} R={score(region_raw)[1]:.2f} F1={score(region_raw)[2]:.2f}  ({len(region_raw)} cuts)")

# sweep prob-only, ssim+density
best=None
def keep_rule(Tp, Ts, kg):
    sel=[c["time"] for c in cands if c["prob"]>=Tp and c["ssim"]<Ts]
    if kg: sel=dampen_strobe(sel, dens_window=8, dens_max=4, merge_gap=4, keep_gap=kg)
    return sel
print("\n=== sweep: prob>=Tp AND ssim<Ts [+ density keep_gap] ===")
for Tp in (0.1,0.2,0.3,0.4):
  for Ts in (0.55,0.6,0.65,0.7,1.0):
    for kg in (0,2.5,3.5,5.0):
      sel=keep_rule(Tp,Ts,kg); p,r,f=score(sel)
      if best is None or f>best[0]: best=(f,p,r,Tp,Ts,kg,len(sel))
f,p,r,Tp,Ts,kg,ncut=best
print(f"  BEST: F1={f:.2f} P={p:.2f} R={r:.2f}  @ prob>={Tp}, ssim<{Ts}, keep_gap={kg}  ({ncut} cuts vs {len(canon)} canonical)")
