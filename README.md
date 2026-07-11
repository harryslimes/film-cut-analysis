# cut-time-extractor

Detect the sequence of **cut times** (shot boundaries) in a video. Several
implementations of the standard "how much did the frame change?" idea, plus a neural
option, with a benchmark harness that scores each on speed *and* accuracy against
labelled ground truth.

The design targets the two false positives you asked to avoid:
- **a light switching on in a dark scene** — handled by comparing frames in HSV and
  down-weighting brightness (Value), which is the only channel a uniform light change moves.
- **a fast whip-pan** — handled by *adaptive* thresholding: a cut must spike above its
  local neighbourhood; a sustained pan is high-but-flat, so it isn't flagged.

## Implementations

| name | how it works | notes |
|---|---|---|
| `ffmpeg-cpu` / `ffmpeg-cuda` | FFmpeg's built-in `scene` metric (normalised whole-frame diff) | fastest to set up; naive, flash-prone |
| `psd-content` | PySceneDetect `ContentDetector` — HSV-weighted diff | robust to light-on |
| `psd-adaptive` | PySceneDetect `AdaptiveDetector` — spike vs rolling neighbourhood | best classic all-rounder for pans |
| `torch-cpu` / `torch-cuda` | our own pipeline: (NVDEC) decode → GPU HSV diff → adaptive threshold | brightness+pan robust, tweakable |
| `transnet-cpu` / `transnet-cuda` | **TransNetV2** neural net on 48×27 thumbnails | best on *real* footage; also catches dissolves/fades |

`-cpu` / `-cuda` refers to the **decode** path (software vs NVDEC); TransNet/torch
compute always runs on the GPU when available.

## Quick start

```bash
# 1. make a labelled synthetic test clip (known cuts + light-on/whip-pan traps)
python make_test_video.py --seconds 90 --fps 30

# 2. compare every detector on speed + accuracy
python benchmark.py data/test.mp4 --tol 0.4

# 3. extract cut times from a real video -> CSV / EDL / JSON
python cut_times.py movie.mkv --detector psd-adaptive --format all
python cut_times.py --list        # see all detector names
```

## Results on the labelled synthetic clip (90 s, 720p)

P = precision, R = recall, traps = false positives on light-on/whip-pan (lower is better):

```
detector          cuts    fps   xRT    P     R    F1  traps
ffmpeg-scene[cpu]   29    1312    44  1.00  0.94  0.97   0
psd-content         31     999    33  1.00  1.00  1.00   0
psd-adaptive        33    1005    34  0.94  1.00  0.97   2
torch-gpu[cpu]      31     860    29  1.00  1.00  1.00   0
transnetv2[cpu]     38     990    33  0.82  1.00  0.90   7
```

**Caveat that matters:** this synthetic clip is a torture test for the *traps*, but it's
**out-of-distribution for TransNetV2**, which is trained on real footage — that's why the
neural net over-fires on the synthetic whip-pans here. On real movies the ranking flips
and TransNetV2 is normally the most accurate. Use synthetic data to probe robustness
behaviour; use **real labelled footage** (see [DATASETS.md](DATASETS.md)) to rank
real-world accuracy.

## On speed (measured on a 10-min 1080p clip)

Full decode of 1080p H.264 runs ~500–800 fps here (≈20–26× realtime → a 2-hour film in
~5–7 min). A non-obvious finding: **single-stream NVDEC + `hwdownload` was *slower* than
16-thread software decode** — copying every frame GPU→CPU over PCIe and colour-converting
on one thread eats the decode saving. The GPU only pulls ahead when you either keep frames
on-device (zero-copy, e.g. PyNvVideoCodec / DALI) **or** decode many videos concurrently
so NVDEC stays saturated while the CPU is free. `concurrency_demo.py` measures that
scaling. For a handful of movies, plain software decode is already fast enough.

> Speed wasn't the priority for this build — correctness was. The numbers above are from
> a single run, not a tuned benchmark.

## Files

- `make_test_video.py` — generate a labelled clip (`data/test.gt.json` is the ground truth)
- `detectors/` — one module per method + a `REGISTRY`
- `benchmark.py` — run all detectors, time them, score vs ground truth
- `evaluate.py` — precision/recall/F1 matching + trap false-positive check
- `cut_times.py` — practical CLI → CSV / EDL / JSON
- `movienet_to_gt.py` — convert a MovieNet/SceneSeg shot list → ground-truth JSON
- `sync.py` — align a reference cut list to YOUR copy of a film (offset + fps scale)
- `concurrency_demo.py` — aggregate throughput vs number of concurrent streams
- `DATASETS.md` — where to get labelled shot-boundary data for real/famous movies

## Recommendation

- **Fast + robust, no extra deps beyond ffmpeg:** `psd-adaptive` or `torch-cpu`.
- **Best accuracy on real movies (you have a 5090):** `transnet-cuda` — also catches
  dissolves/fades the threshold methods miss.
- Detect at low resolution (already done: ~180 px tall for torch, 48×27 for TransNet) —
  cut detection doesn't need full res, and decode is the bottleneck.
