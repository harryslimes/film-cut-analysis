"""Cut detection from the encoder's own motion vectors — no pixel analysis.

Instead of *computing* how much the frame changed (SAD / HSV diff / optical flow),
this reads the decisions the H.264/HEVC encoder already baked into the bitstream:

  * where it inserted **I-frames** (an encoder's scene-cut detector fires here), and
  * the **motion vectors** it stored for P/B macroblocks (via `flags2 +export_mvs`).

A shot boundary shows up as an I-frame that the encoder inserted *because the content
broke*, not because a keyframe was merely due. We separate those two using signals that
are free once the frame is decoded:

  earliness   the I-frame fired well *before* the forced-keyframe deadline (keyint).
              A routine periodic keyframe sits right at the deadline; a scene-cut one
              fires early because x264's scenecut metric tripped.
  corrob      a local motion-vector anomaly straddling the frame: a B-frame that predicts
              almost entirely from one side (its other reference is across the cut), or a
              P-frame whose inter coverage collapses (blocks go intra — nothing to predict
              from). Present on some encodes, absent on others, so it only *adds* evidence.

score = corrob + 0.5 * earliness ;  keep I-frames scoring >= `keep`.

This is decoder-cheap: no optical flow, no NN, no HSV — just the side-data ffmpeg hands
back. It runs ~150-220x realtime here (vs ~30x for the pixel/NN detectors). The catch is
it inherits the encoder's choices: a **fixed-GOP re-encode** (every I-frame periodic, no
scene-cut keyframes) carries no boundary information, so we flag that case in `extra`.

Needs PyAV (`pip install av`). Decode is CPU-side; `method` is accepted for interface
parity but ignored (there is no NN/GPU compute stage here).
"""
from __future__ import annotations

import numpy as np

from .base import CutEvent, Confidence, DetectResult, Timer, ffprobe_info


def _extract(video_path):
    """Single decode pass -> per-frame (t, pict_type, cov, fwd, bwd).

    cov = fraction of frame area covered by inter (motion-compensated) blocks;
    fwd/bwd = the share predicted from a past / future reference. Intra and I-frame
    blocks contribute no motion vector, so low cov == high intra.
    """
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.codec_context.options = {"flags2": "+export_mvs"}
    W = stream.codec_context.width
    H = stream.codec_context.height
    area = float(W * H)

    t, ty, cov, fwd, bwd = [], [], [], [], []
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        t.append(float(frame.pts * stream.time_base))
        ty.append(frame.pict_type.name)
        sd = frame.side_data.get("MOTION_VECTORS")
        if sd is None:
            cov.append(0.0); fwd.append(0.0); bwd.append(0.0)
            continue
        a = sd.to_ndarray()
        blk = a["w"].astype(np.int64) * a["h"].astype(np.int64)
        src = a["source"]                       # <0 past ref, >0 future ref
        f = float(blk[src < 0].sum()) / area
        b = float(blk[src > 0].sum()) / area
        fwd.append(f); bwd.append(b); cov.append(f + b)
    container.close()
    return (np.array(t), np.array(ty), np.array(cov),
            np.array(fwd), np.array(bwd), W, H)


def detect(video_path, keep=0.3, earliness_w=0.5, method="cpu") -> DetectResult:
    fps, n_frames, _, _ = ffprobe_info(video_path)

    with Timer() as timer:
        t, ty, cov, fwd, bwd, W, H = _extract(video_path)
        n = len(t)
        isI, isB, isP = ty == "I", ty == "B", ty == "P"
        # backward share of a B-frame's motion; 0.5 == balanced (no boundary nearby)
        bshare = np.where(cov > 0, bwd / np.maximum(cov, 1e-9), 0.5)

        # local baseline of P-frame inter-coverage (rolling median over ~50 P-frames)
        Pidx = np.where(isP)[0]
        base = np.full(n, np.nan)
        if len(Pidx):
            pcov = cov[Pidx]
            for k, i in enumerate(Pidx):
                lo, hi = max(0, k - 25), min(len(Pidx), k + 25)
                base[i] = np.median(pcov[lo:hi])

        Iidx = np.where(isI)[0]
        It = t[Iidx]
        events, extra = [], {}
        if len(It) >= 3:
            diffs = np.diff(It)
            keyint = float(np.percentile(diffs, 99)) or float(diffs.max())
            gaps = np.concatenate([[keyint], diffs])
            # fraction of I-frames sitting at the same (forced) spacing -> fixed GOP
            regularity = float(np.mean(np.abs(diffs - np.median(diffs)) < 0.05))
            earliness = np.clip(1.0 - gaps / keyint, 0.0, 1.0)

            def corrob(i):
                s = 0.0
                for j in range(max(0, i - 2), min(n, i + 4)):
                    if isB[j] and cov[j] > 0.05:
                        s = max(s, abs(bshare[j] - 0.5) * 2)
                    elif isP[j] and not np.isnan(base[j]) and base[j] > 0 and cov[j] < base[j]:
                        s = max(s, 1 - cov[j] / base[j])
                return s

            corr = np.array([corrob(i) for i in Iidx])
            score = corr + earliness_w * earliness
            # keep-point: an I-frame is a cut when its combined score clears `keep`.
            # Build events here so each carries frame + native mv_score. time is the
            # I-frame's PTS timestamp (round(It[k], 3)) exactly as before, and events are
            # sorted by time, so the .cuts projection is byte-identical by construction.
            # frame = Iidx[k], the I-frame's index in the detector's decoded frame
            # sequence; time comes from PTS, so frame/fps != time here (§3.2, expected).
            # transition_kind stays "unknown": this scores whether an I-frame is a scene
            # cut, with no hard/gradual classifier (§3.2, never a guessed "hard").
            kept = [(round(float(It[k]), 3), int(Iidx[k]), float(score[k]))
                    for k in range(len(Iidx)) if score[k] >= keep]
            kept.sort(key=lambda r: r[0])
            events = [CutEvent(time=tm, frame=fr,
                               confidence=Confidence(sc, "mv_score", higher_is_stronger=True))
                      for tm, fr, sc in kept]
            extra = {"keyint": round(keyint, 2), "regularity": round(regularity, 2),
                     "n_iframes": int(isI.sum()), "fixed_gop": regularity > 0.9}
            if extra["fixed_gop"]:
                extra["warning"] = ("fixed-GOP encode: I-frames are periodic, not "
                                    "scene-cuts; motion-vector cuts are unreliable here")

    return DetectResult(
        name="motion-vectors", events=events, elapsed=timer.elapsed,
        n_frames=n_frames, fps_source=fps, extra=extra,
    )
