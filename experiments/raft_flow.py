"""Test RAFT optical-flow continuity for flash-vs-cut on Vertigo's nightmare.

For each detected cut: RAFT flow between the before/after frames, warp before->after,
and measure the residual on CLAHE-normalised grayscale (strips the colour flash). If the
after-frame is a flow-transform of the before-frame (a flash or coherent motion), residual
is LOW; a real cut can't be explained by any flow -> residual HIGH. Renders a decision
sheet so we can SEE whether flow beats SSIM on the graveyard/green cases.
"""
import json, subprocess, argparse
import numpy as np, cv2, torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--cuts", required=True)
ap.add_argument("--start", type=float, required=True); ap.add_argument("--end", type=float, required=True)
ap.add_argument("--fps", type=float, default=23.976); ap.add_argument("--delta", type=float, default=0.2)
ap.add_argument("--thresh", type=float, default=18.0); ap.add_argument("--out", required=True)
ap.add_argument("--tw", type=int, default=160); ap.add_argument("--cols", type=int, default=4)
a = ap.parse_args()
a0 = a.start - 1.0
RW, RH = 256, 144  # divisible by 8 for RAFT

def dec(sc): return subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-ss",str(a0),
    "-t",str(a.end-a0+1),"-i",a.video,"-vf",sc,"-f","rawvideo","-"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
gray = np.frombuffer(dec(f"scale={RW}:{RH},format=gray"), np.uint8).reshape([-1, RH, RW])
pr = subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
    "stream=width,height","-of","csv=p=0",a.video]).decode().strip().split(",")
W,H=int(pr[0]),int(pr[1]); th=int(round(a.tw*H/W/2)*2)
col = np.frombuffer(dec(f"scale={a.tw}:{th},format=bgr24"), np.uint8).reshape([-1, th, a.tw, 3])
clahe = cv2.createCLAHE(2.0, (8, 8)); grayc = np.stack([clahe.apply(f) for f in gray])

cuts = [c for c in json.load(open(a.cuts))["cuts"] if a.start <= c <= a.end]
d = int(round(a.delta * a.fps))
dev = "cuda"
weights = Raft_Large_Weights.DEFAULT
model = raft_large(weights=weights, progress=False).eval().to(dev)
tf = weights.transforms()

def to3(f):  # gray uint8 HxW -> 3xHxW uint8 tensor
    return torch.from_numpy(np.repeat(f[None], 3, 0))

befs = torch.stack([to3(grayc[max(0, int(round((c-a0)*a.fps))-d)]) for c in cuts])
afts = torch.stack([to3(grayc[min(len(grayc)-1, int(round((c-a0)*a.fps))+d)]) for c in cuts])
resid = []
with torch.no_grad():
    for i in range(0, len(cuts), 16):
        b, f = befs[i:i+16].to(dev), afts[i:i+16].to(dev)
        b1, f1 = tf(b, f)
        flow = model(b1, f1)[-1]                      # (n,2,H,W)
        n, _, Hh, Ww = flow.shape
        yy, xx = torch.meshgrid(torch.arange(Hh, device=dev), torch.arange(Ww, device=dev), indexing="ij")
        gx = (xx[None] + flow[:, 0]) / (Ww - 1) * 2 - 1
        gy = (yy[None] + flow[:, 1]) / (Hh - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1)
        # friend's metric: flow-field CHAOS = spatial smoothness of the flow.
        # coherent motion (pan/rotation) -> smooth field -> low; real cut -> chaotic -> high
        dfx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
        dfy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
        resid.extend((dfx + dfy).cpu().tolist())

# render decision sheet: keep if residual >= thresh (real cut), drop if low (flash/motion)
pad,lab=8,20; pair_w=a.tw*2+pad; cw=pair_w+pad*2; ch=th+lab+pad*2
cols=a.cols; rows=(len(cuts)+cols-1)//cols
sheet=np.full((rows*ch+pad, cols*cw+pad,3),30,np.uint8); nk=0
for k,c in enumerate(cuts):
    ig=int(round((c-a0)*a.fps)); keep = resid[k] >= a.thresh; nk += keep
    r,cc=divmod(k,cols); y=pad+r*ch+lab; x=pad+cc*cw
    sheet[y:y+th,x:x+a.tw]=col[max(0,ig-d)]; sheet[y:y+th,x+a.tw+pad:x+2*a.tw+pad]=col[min(len(col)-1,ig+d)]
    color=(0,220,0) if keep else (0,0,230)
    cv2.rectangle(sheet,(x-2,y-2),(x+pair_w+2,y+th+2),color,2)
    cv2.putText(sheet,f"#{k+1} {c:.1f} r{resid[k]:.0f} {'KEEP' if keep else 'drop'}",
                (x,pad+r*ch+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1,cv2.LINE_AA)
cv2.imwrite(a.out,sheet)
print(f"{len(cuts)} cuts -> keep {nk}, drop {len(cuts)-nk} (RAFT residual>={a.thresh}) -> {a.out}")
