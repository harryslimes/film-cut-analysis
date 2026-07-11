# Labelled shot-boundary data (for training / evaluation)

**Yes, real full-length movies have downloadable cut data.** The one catch: the datasets
give you the **shot list (frame-accurate cut points), not the movie file** — you supply
your own copy of the film and match by frame number × fps. That's the standard, legal
arrangement, and it's all you need to validate a detector. Use
[movienet_to_gt.py](movienet_to_gt.py) to turn any of these shot lists into the ground-
truth JSON this repo's benchmark reads.

## ⭐ Real full-length feature films, frame-accurate cuts

| Source | Coverage | What you get | Access |
|---|---|---|---|
| **MovieNet — `movie1K.shot_detection.v1`** | **1,100 real movies** (Titanic, Inception, …) | one text file per film: every shot's start/end frame | Free account at OpenDataLab: `opendatalab.com/OpenDataLab/MovieNet` (20.9 MB zip). The old direct `download.openmmlab.com/datasets/movienet/movie1K.shot_detection.v1.zip` is now deprecated. |
| **MovieNet-318 / SceneSeg** (AnyiRao, CVPR2020) | **318 movies** | shot boundaries + scene labels; also re-hosted by **BaSSL** (kakaobrain/bassl) as `shot_movie318` with a download script | `github.com/AnyiRao/SceneSeg` (INSTALL.md links) / `github.com/kakaobrain/bassl` |
| **Cinemetrics** (Univ. of Chicago) | **thousands** of named films, hand-timed | per-shot cut *timings* (advanced/timed mode), exportable as `.cmsx` | `cinemetrics.uchicago.edu/database` — browse, filter to "advanced" entries, Export. Crowd-sourced → messy, ~0.3 s manual timing error. |
| **SoccerNet-v2** | 500 full matches (~1.5 h each) | 158,493 camera shot boundaries | not movies, but full-length real broadcast with dense labels |

**How to use MovieNet/SceneSeg to validate on a real film:**
```bash
# 1. download the shot annotation for your movie from OpenDataLab (e.g. shot_tt1375666.txt)
# 2. convert to this repo's ground-truth format (supply the REFERENCE fps)
python movienet_to_gt.py shot_tt1375666.txt --fps 23.976 --out data/inception.ref.json
# 3. SYNC the reference cuts onto YOUR copy of the film (handles logo offset + PAL speed)
python sync.py --ref data/inception.ref.json --video inception.mkv --out data/inception.gt.json
# 4. score your detector against the synced ground truth
python benchmark.py inception.mkv --only transnet-cuda,psd-adaptive --tol 0.4
```

Your rip won't line up with the reference encode — it has studio-logo/leader offset and
maybe a different frame rate (PAL 25 fps is 4 % faster). [sync.py](sync.py) recovers the
affine map `your_time = scale·ref_time + offset` by aligning the two cut *patterns*, and
reports a match rate that tells you if it's the same version or a different cut.

## Immediately loadable (no account, includes real-movie footage)

| Source | What it is | Load |
|---|---|---|
| **HuggingFace `it-just-works/shot-boundary-detection`** | AutoShot + ClipShots + Pexels, real & synthetic, 61-frame labelled clips, MIT | `datasets.load_dataset("it-just-works/shot-boundary-detection")` |
| **AutoShot / SHOT** (2023) | 853 short videos, 11,606 shot annotations | `arxiv.org/abs/2304.06116` + repo |

## Standard train / test sets for cut detection

> Verify licence before use; some need a research-use agreement.

## Best for "famous movies with shot labels"

| Dataset | What it is | Size | Use |
|---|---|---|---|
| **MovieNet** (CUHK) | 1,100 real movies with shot boundaries + scene/char/action labels | ~1.6M shots | Closest to "famous movies, labelled". Research licence; distributed as keyframes/features/clips, not full films. `movienet.github.io` |
| **Cinemetrics** | Human-coded shot lengths (ASL / per-shot timings) for real films | ~17k films | Validate average-shot-length & distributions against real cinema. `cinemetrics.lv` |

## Standard train / test sets for cut detection

| Dataset | What it is | Role |
|---|---|---|
| **ClipShots** | 4,039 short web videos, ~128k annotated cuts + gradual transitions | The main **training** set for TransNetV2. `github.com/Tangshitao/ClipShots` |
| **BBC Planet Earth** (Baraldi et al. 2015) | 11 documentary episodes, manual shot boundaries | Common **test** set; "famous footage". UNIMORE imagelab |
| **RAI** | 10 Italian broadcast videos with shot annotations | Small **test** set. UNIMORE imagelab |
| **TRECVID SBD 2001–2007** | The classic NIST shot-boundary benchmark | Historical gold standard; videos via NIST, annotations public |
| **AutoShot / SHOT** (2023) | 853 short clips, modern annotations | Newer benchmark, harder transitions |

The TransNetV2 recipe (what the bundled weights were trained on): **train on ClipShots +
synthetically-generated transitions, test on BBC/RAI**. The synthetic part is the same
trick as our [make_test_video.py](make_test_video.py) — compose known cuts/dissolves from
clips so you get free frame-accurate labels.

## Cheapest path to your own labelled set

You do **not** need to hand-label from scratch:

1. Run a strong detector (TransNetV2, or `psd-adaptive`) to get *candidate* cuts.
2. Load the candidates + video into a player and hand-correct — fix misses/false-alarms.
   PySceneDetect can emit a CSV of candidates and even split the video at each cut so you
   can eyeball them fast.
3. Save corrected cuts as ground truth (same JSON schema as `data/test.gt.json`).

Semi-automatic labelling like this is ~10× faster than scrubbing frame by frame, and a
couple of your own movies is enough to *validate* accuracy (you don't need thousands
unless you're training a network from scratch — for that, start from ClipShots).

## Evaluation protocol used here

`evaluate.py` does greedy 1-to-1 matching of predicted cuts to ground-truth cuts within a
temporal tolerance (default ±0.4–0.5 s), then reports precision / recall / F1 — the same
metric family as the TRECVID/TransNetV2 evaluations. `trap_hits` additionally counts
false alarms that land inside a deliberately hard shot (light-on / whip-pan).
