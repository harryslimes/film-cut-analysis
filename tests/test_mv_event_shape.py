"""S3-E5: the enriched event shape motion_vectors.py now emits (design §3.2,
§5 S3-E5). av/numpy-free: motion_vectors.py does `import numpy` at module top
(and `import av` inside _extract), so we reproduce its keep-point construction
with synthetic values -- no av, no numpy, no video. The real import/runtime path
is source-reviewed only.

Per kept I-frame k (score[k] >= keep), motion_vectors builds:
    CutEvent(time=round(float(It[k]), 3), frame=int(Iidx[k]),
             confidence=Confidence(float(score[k]), "mv_score", higher_is_stronger=True))
transition_kind defaults "unknown"; no span, no flags. score = corr + 0.5*earliness
is the exact quantity compared to `keep`. time is the I-frame's PTS, NOT frame/fps.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cut_events import (  # noqa: E402
    Confidence, CutEvent, event_to_dict, validate_events,
)


def _mv_events(iframes, keep):
    """Reproduce motion_vectors.py's keep-point: filter by score, sort by time.
    iframes: list of (pts_seconds, iframe_index, score)."""
    kept = [(round(float(pts), 3), int(idx), float(sc))
            for (pts, idx, sc) in iframes if sc >= keep]
    kept.sort(key=lambda r: r[0])
    return [CutEvent(time=tm, frame=fr,
                     confidence=Confidence(s, "mv_score", higher_is_stronger=True))
            for tm, fr, s in kept]


class TestMvEventShape(unittest.TestCase):
    def test_kept_event_shape(self):
        (e,) = _mv_events([(12.3456, 296, 0.8)], keep=0.3)
        self.assertEqual(e.confidence.metric, "mv_score")
        self.assertEqual(e.confidence.value, 0.8)          # native passthrough
        self.assertTrue(e.confidence.higher_is_stronger)
        self.assertEqual(e.transition_kind, "unknown")     # no classifier -> never "hard"
        self.assertIsInstance(e.frame, int)
        self.assertEqual(e.frame, 296)
        self.assertEqual(e.time, round(12.3456, 3))        # 3-dp PTS rounding, as the detector does
        self.assertIsNone(e.span)
        self.assertIsNone(e.flags)

    def test_native_value_passthrough_can_exceed_one(self):
        # score = corr(1.0) + 0.5*earliness(0.7) = 1.35 -> stored raw, NOT clamped to [0,1]
        (e,) = _mv_events([(1.0, 24, 1.35)], keep=0.3)
        self.assertEqual(e.confidence.value, 1.35)
        self.assertGreater(e.confidence.value, 1.0)

    def test_frame_is_iframe_index_time_is_pts_not_frame_over_fps(self):
        # time comes from PTS; frame is the I-frame index -> frame/fps != time (§3.2)
        (e,) = _mv_events([(2.0, 61, 0.5)], keep=0.3)
        self.assertEqual(e.time, 2.0)
        self.assertEqual(e.frame, 61)
        self.assertNotAlmostEqual(e.frame / 24.0, e.time)   # 61/24 = 2.5417 != 2.0

    def test_keep_filter_and_sort_projection(self):
        # below-threshold I-frames dropped; survivors sorted by time; projection matches
        iframes = [(30.0, 900, 0.10),   # dropped (0.10 < 0.3)
                   (10.0, 240, 0.55),   # kept
                   (20.0, 480, 0.30),   # kept (== keep)
                   (5.0, 120, 0.20)]    # dropped
        evs = _mv_events(iframes, keep=0.3)
        self.assertEqual([e.time for e in evs], [10.0, 20.0])       # sorted, filtered
        self.assertEqual([e.frame for e in evs], [240, 480])
        validate_events(evs)

    def test_no_span_no_flags_minimal_keys(self):
        (e,) = _mv_events([(7.0, 168, 0.9)], keep=0.3)
        d = event_to_dict(e)
        self.assertEqual(set(d), {"time", "frame", "transition_kind", "confidence"})
        self.assertNotIn("span", d)
        self.assertNotIn("flags", d)


if __name__ == "__main__":
    unittest.main()
