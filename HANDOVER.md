# Handover — cut-time-extractor

Detect the sequence of **cut times** (shot boundaries) in a video, score against real
labels, and produce shot-counter subtitles. Built and validated on a real RTX 5090.

## TL;DR state
- **Works well.** Six real feature films scored vs MovieNet ground truth: **mean F1 ≈ 0.93**
  (TransNetV2, defaults tuned on data). See `results/batch_scores.json`.
- Detectors, sync, batch scoring, a labelling web tool, and per-film SRTs all functional.
- One genuinely hard sequence (Vertigo's "Nightmare", 5030–5115 s) is at the **automation
  ceiling (~F1 0.55)** — handled with a **human-verified canonical** (see below).

## Pipeline (core files)
- `detectors/` — detector registry. Best: `transnet-cuda` (TransNetV2). Also `psd-*`
  (PySceneDetect), `ffmpeg-*`, `torch-*` (custom HSV), and `motion-vectors` (in progress,
  another agent — reads H.26x motion vectors / I-frames via PyAV).
  - `transnet.py` — TransNetV2 wrapper. Tuned defaults: `threshold=0.40` (sharp cuts),
    `gradual_height=0.35` (dissolve peaks from the all-frames stream), `strobe_guard=True`.
- `cut_times.py` — CLI: video → cuts as CSV / EDL / **SRT** (shot counter) / JSON.
- `sync.py` — aligns a reference cut list onto your copy (recovers fps-scale + leader
  offset by aligning the cut *pattern*). Handles PAL/leader mismatches.
- `movienet_to_gt.py` — MovieNet shot files → reference cut list.
- `batch_score.py` — scores every movie under `data/movies/` vs its MovieNet ground truth,
  **in parallel** (`--workers N`, ~4× at N=8; decode-bound, NVDEC scales to ~5×).
- `postfilter.py` — `dampen_strobe` (density collapse) + `collapse_region` (targeted).
- `structural.py` — CLAHE-SSIM flash rejection for dense regions.
- `evaluate.py` — precision/recall/F1 matching with a temporal tolerance.
- `make_test_video.py` / `make_srt.py` / `benchmark.py` — synthetic labelled clip, SRT from
  any cut list, and the detector comparison harness.

## Tuning tools (kept in root)
- `tune_transnet.py`, `tune_gradual.py` — sweep thresholds vs ground truth.
- `hf_validate.py` — validate thresholds across the HuggingFace shot-boundary clip set.
- `experiments/` — the Vertigo-nightmare investigation (contact sheets, SSIM/edge/RAFT
  tests, decision renderers). Kept for reference; not part of the pipeline.

## Ground truth / canonical (IMPORTANT for further testing)
- `groundtruth/vertigo_nightmare.canonical.json` — **9 human-verified cuts** for the
  Nightmare sequence (5030–5115 s), made with the labelling tool. This is the canonical
  reference for that sequence. **Any detector (incl. the motion-vector one) should be
  scored against this** for the hard sequence:
  ```bash
  python score_canonical.py --gt groundtruth/vertigo_nightmare.canonical.json \
      --video "<Vertigo mkv>" --detector motion-vectors --tol 0.4
  ```
- MovieNet shot labels for the 6 films live under `data/movienet/` (not committed; pulled
  via OpenDataLab — see DATASETS.md). Films: Charade, His Girl Friday (not in MovieNet),
  Wizard of Oz, It's A Wonderful Life, Rear Window, Vertigo, North by Northwest.

## Labelling tool (human-in-the-loop)
- `label_prep.py` — run TransNet on a span, extract all candidates (both streams) + before/
  after thumbnails, mark current keeps.
- `label_server.py` — local web tool (`http://localhost:8770`): reject wrong keeps, expand
  gaps to surface lower-confidence candidates (time-ordered), promote missed cuts, SAVE →
  `canonical.json` + `canonical.srt`. This is how the Vertigo canonical was made.

## The Vertigo Nightmare saga (why it's canonical, not automated)
Saul Bass's animated spiral + colour flashes over Scotty's face. Tested exhaustively:
raw pixels, TransNetV2, grayscale edges, Farneback flow, **RAFT** (residual AND coherence),
CLAHE-SSIM, temporal-median windows. **Conclusion:** real cuts and flashes are
indistinguishable on every frame-level signal (your 9 real cuts span prob 0.18–0.98; flashes
span 0.11–0.99 — total overlap). Best auto rule = F1 0.55. Also your canonical encodes a
*semantic* call (the 5-frame flower montage = ONE cut), which no signal can infer. So: use
the canonical. The rest of Vertigo and all other films auto-score 0.89–0.95.

## Open threads / next steps
1. **Motion-vector detector** (another agent, `detectors/motion_vectors.py`, `mv_benchmark.py`)
   — reads the encoder's own MVs from the H.265 file. Score it against
   `groundtruth/vertigo_nightmare.canonical.json`. Interesting question: do MVs separate
   flashes from cuts where pixel signals can't? (A flash → MVs stay coherent; a cut → MV
   field collapses. Worth measuring against the canonical.)
2. **Decision:** wire the CLAHE-SSIM + density hybrid in as the *general* dense-region
   handler (auto-cleans any film's flash montages) vs keeping it Vertigo-scoped. It costs a
   decode of dense regions only. Currently the global default is the cheap `dampen_strobe`.
3. **More canonicals:** label other hard sequences with `label_prep`/`label_server` →
   `groundtruth/`, and add a `--gt` path to `batch_score.py` to fold region canonicals into
   the film score.
4. Residuals we accepted on Vertigo: the flower graphics (semantic) and ~2 graveyard flashes.

## Environment
RTX 5090, ffmpeg with NVDEC (`h264_cuvid`, `scale_cuda`), torch+torchvision CUDA, opencv,
scenedetect, transnetv2-pytorch. `data/` and `results/` are gitignored (large / regenerable).
