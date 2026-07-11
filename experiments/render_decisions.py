"""Contact sheet with KEEP/DROP decision per cut from CLAHE-SSIM (flash = high similarity).
Green border = kept (real cut), red = dropped (flash). Lets us verify the rule visually."""
import json, subprocess, argparse
import numpy as np
import cv2


def ssim(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    ma = cv2.GaussianBlur(a, (7, 7), 1.5); mb = cv2.GaussianBlur(b, (7, 7), 1.5)
    va = cv2.GaussianBlur(a*a, (7, 7), 1.5) - ma**2; vb = cv2.GaussianBlur(b*b, (7, 7), 1.5) - mb**2
    vab = cv2.GaussianBlur(a*b, (7, 7), 1.5) - ma*mb
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    return float((((2*ma*mb+c1)*(2*vab+c2))/((ma**2+mb**2+c1)*(va+vb+c2))).mean())


ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--cuts", required=True)
ap.add_argument("--start", type=float, required=True); ap.add_argument("--end", type=float, required=True)
ap.add_argument("--fps", type=float, default=23.976); ap.add_argument("--delta", type=float, default=0.2)
ap.add_argument("--thresh", type=float, default=0.68); ap.add_argument("--out", required=True)
ap.add_argument("--tw", type=int, default=160); ap.add_argument("--cols", type=int, default=4)
a = ap.parse_args()

a0 = a.start - 1.0
def decode(vf, scale):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(a0), "-t", str(a.end-a0+1),
           "-i", vf, "-vf", scale, "-f", "rawvideo", "-"]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
gray = np.frombuffer(decode(a.video, "scale=256:144,format=gray"), np.uint8).reshape([-1, 144, 256])
probe = subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
    "stream=width,height","-of","csv=p=0",a.video]).decode().strip().split(",")
W, H = int(probe[0]), int(probe[1]); th = int(round(a.tw*H/W/2)*2)
col = np.frombuffer(decode(a.video, f"scale={a.tw}:{th},format=bgr24"), np.uint8).reshape([-1, th, a.tw, 3])
clahe = cv2.createCLAHE(2.0, (8, 8)); grayc = np.stack([clahe.apply(f) for f in gray])

cuts = [c for c in json.load(open(a.cuts))["cuts"] if a.start <= c <= a.end]
d = int(round(a.delta*a.fps)); dg = int(round(a.delta*a.fps))
pad, lab = 8, 20; pair_w = a.tw*2+pad; cw = pair_w+pad*2; ch = th+lab+pad*2
cols = a.cols; rows = (len(cuts)+cols-1)//cols
sheet = np.full((rows*ch+pad, cols*cw+pad, 3), 30, np.uint8)
kept = 0
for k, c in enumerate(cuts):
    ig = int(round((c-a0)*a.fps))
    s = ssim(grayc[max(0,ig-dg)], grayc[min(len(grayc)-1,ig+dg)])
    keep = s < a.thresh
    kept += keep
    r, cc = divmod(k, cols); y = pad+r*ch+lab; x = pad+cc*cw
    ib, ia = max(0,ig-d), min(len(col)-1, ig+d)
    sheet[y:y+th, x:x+a.tw] = col[ib]; sheet[y:y+th, x+a.tw+pad:x+2*a.tw+pad] = col[ia]
    color = (0,220,0) if keep else (0,0,230)
    cv2.rectangle(sheet, (x-2, y-2), (x+pair_w+2, y+th+2), color, 2)
    cv2.putText(sheet, f"#{k+1} {c:.1f} ssim{s:.2f} {'KEEP' if keep else 'drop'}",
                (x, pad+r*ch+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
cv2.imwrite(a.out, sheet)
print(f"{len(cuts)} cuts -> keep {kept}, drop {len(cuts)-kept} (thresh ssim<{a.thresh}) -> {a.out}")
