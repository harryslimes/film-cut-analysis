"""For each detected cut in a span, compute illumination-invariant structural-similarity
scores between the before/after frames and print them, so we can see (and measure) which
score separates colour-flashes (same composition -> high similarity) from real cuts
(different composition -> low similarity). No thresholding claims -- just the numbers.
"""
import json, subprocess, argparse
import numpy as np
import cv2


def ssim(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    mu_a = cv2.GaussianBlur(a, (7, 7), 1.5); mu_b = cv2.GaussianBlur(b, (7, 7), 1.5)
    va = cv2.GaussianBlur(a * a, (7, 7), 1.5) - mu_a ** 2
    vb = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mu_b ** 2
    vab = cv2.GaussianBlur(a * b, (7, 7), 1.5) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def flow_residual(a, b):
    """Warp a->b by optical flow; low residual => b is a is same scene (just moved/relit)."""
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    h, w = a.shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mapx = (gx + flow[..., 0]).astype(np.float32); mapy = (gy + flow[..., 1]).astype(np.float32)
    warp = cv2.remap(a, mapx, mapy, cv2.INTER_LINEAR)
    # compare structure after CLAHE so colour-flash brightness is neutralised
    return float(np.abs(warp.astype(np.float32) - b.astype(np.float32)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--cuts", required=True)
    ap.add_argument("--start", type=float, required=True); ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--fps", type=float, default=23.976); ap.add_argument("--delta", type=float, default=0.2)
    args = ap.parse_args()
    a0 = args.start - 1.0
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(a0), "-t", str(args.end - a0 + 1),
           "-i", args.video, "-vf", "scale=256:144,format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    F = np.frombuffer(raw, np.uint8).reshape([-1, 144, 256])
    clahe = cv2.createCLAHE(2.0, (8, 8))
    Fc = np.stack([clahe.apply(f) for f in F])   # local-contrast normalised (kills colour flash)

    cuts = [c for c in json.load(open(args.cuts))["cuts"] if args.start <= c <= args.end]
    d = int(round(args.delta * args.fps))
    print(f"{'#':>3} {'time':>8} {'ssim_clahe':>10} {'flow_resid':>10}")
    for k, c in enumerate(cuts, 1):
        i = int(round((c - a0) * args.fps))
        ib, ia = max(0, i - d), min(len(F) - 1, i + d)
        s = ssim(Fc[ib], Fc[ia])
        fr = flow_residual(Fc[ib], Fc[ia])
        print(f"{k:3d} {c:8.1f} {s:10.3f} {fr:10.1f}")


if __name__ == "__main__":
    main()
