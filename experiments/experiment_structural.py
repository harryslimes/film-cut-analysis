"""Test the 'structural continuity' hypothesis on Vertigo's nightmare.

Does an edge/structure-based signal stay LOW through flashes (structure preserved) but
SPIKE at real cuts (structure replaced), where raw pixel diff can't tell them apart?

Decodes the region to greyscale and, for each consecutive frame pair, computes:
  d_raw   : mean |f2 - f1|                        (baseline; fooled by flashes)
  d_norm  : same but each frame z-normalised first (cancels a global brightness flash)
  d_edge  : mean |sobel(f2) - sobel(f1)|          (the wireframe idea, motion-sensitive)
  d_edgeMC: edge diff, min over small shifts       (poor-man's motion compensation)
Then prints these at each TransNet-detected cut so we can see which are 'structural'.
"""
import json, subprocess
import numpy as np
import cv2

V = "data/movies/Vertigo.1958.1080p.CRIT.WEB-DL.AAC2.0.H.264-DonK/Vertigo (1958) (1080p CRIT WEB-DL H264 SDR AAC 2.0 English - DonK)"
A, B, FPS = 5030.0, 5115.0, 23.976
W, H = 256, 144

cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(A), "-t", str(B - A),
       "-i", V + ".mkv", "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"]
raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
F = np.frombuffer(raw, np.uint8).reshape([-1, H, W]).astype(np.float32)
n = len(F)

# precompute normalised frames and edge maps
Fz = np.stack([(f - f.mean()) / (f.std() + 1e-3) for f in F])
E = np.stack([cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3),
                            cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)) for f in F])
Ez = np.stack([(e / (e.mean() + 1e-3)) for e in E])   # normalise edge energy too

def d_raw(i):  return np.abs(F[i] - F[i-1]).mean()
def d_norm(i): return np.abs(Fz[i] - Fz[i-1]).mean()
def d_edge(i): return np.abs(Ez[i] - Ez[i-1]).mean()
def d_edge_mc(i):  # min edge-diff over +-4px shifts (tolerate motion)
    best = 1e9
    for dy in (-4, 0, 4):
        for dx in (-4, 0, 4):
            a = np.roll(np.roll(Ez[i], dy, 0), dx, 1)
            best = min(best, np.abs(a - Ez[i-1]).mean())
    return best

cuts = [c for c in json.load(open(V + ".cuts.json"))["cuts"] if A < c < B]
# normalise each signal to its own median so they're comparable
sig = {name: np.array([fn(i) for i in range(1, n)]) for name, fn in
       [("raw", d_raw), ("norm", d_norm), ("edge", d_edge), ("edgeMC", d_edge_mc)]}
med = {k: np.median(v) for k, v in sig.items()}

print(f"region {A}-{B}s  {n} frames  {len(cuts)} detected cuts")
print("signal values at each detected cut (x its own median; high = discontinuity):")
print(f"{'time':>8} {'raw':>6} {'norm':>6} {'edge':>6} {'edgeMC':>7}")
for c in cuts:
    i = int(round((c - A) * FPS))
    i = max(1, min(n - 1, i))
    # take the max over a small window around the cut
    w = range(max(1, i-1), min(n, i+2))
    vals = {k: max(sig[k][j-1] for j in w) / med[k] for k in sig}
    print(f"{c:8.2f} {vals['raw']:6.1f} {vals['norm']:6.1f} {vals['edge']:6.1f} {vals['edgeMC']:7.1f}")
