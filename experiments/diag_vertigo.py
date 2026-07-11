"""Diagnostic: in Vertigo's nightmare, do real cuts have stronger predictions than flashes?
Decodes the film, and for each detected cut in the region prints its single-frame and
all-frames prediction strength, and whether a MovieNet GT cut is nearby."""
import json, subprocess
import numpy as np, torch
from detectors.base import ffprobe_info
from detectors.transnet import _model

V = "data/movies/Vertigo.1958.1080p.CRIT.WEB-DL.AAC2.0.H.264-DonK/Vertigo (1958) (1080p CRIT WEB-DL H264 SDR AAC 2.0 English - DonK)"
A, B = 5030, 5110

fps, n, W, H = ffprobe_info(V + ".mkv")
cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda",
       "-hwaccel_output_format", "cuda", "-i", V + ".mkv",
       "-vf", "scale_cuda=48:27,hwdownload,format=nv12", "-pix_fmt", "rgb24",
       "-f", "rawvideo", "-"]
raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
frames = np.frombuffer(raw, np.uint8).reshape([-1, 27, 48, 3])
m = _model()
fr = torch.from_numpy(np.ascontiguousarray(frames)).to(m.device)
with torch.no_grad():
    single, allf = m.predict_frames(fr, quiet=True)
single = single.cpu().numpy().reshape(-1); allf = allf.cpu().numpy().reshape(-1)

cuts = [c for c in json.load(open(V + ".cuts.json"))["cuts"] if A <= c <= B]
gt = [c for c in json.load(open(V + ".gt.json"))["cuts"] if A <= c <= B]
print(f"{len(cuts)} detected, {len(gt)} MovieNet-GT cuts in {A}-{B}s\n")
print(f"{'time':>8} {'single':>7} {'all':>6} {'nearGT':>7}")
for c in cuts:
    f = int(round(c * fps))
    s = single[max(0,f-1):f+2].max()
    a = allf[max(0,f-1):f+2].max()
    ng = "yes" if any(abs(c-g) <= 0.5 for g in gt) else ""
    print(f"{c:8.2f} {s:7.3f} {a:6.3f} {ng:>7}")
