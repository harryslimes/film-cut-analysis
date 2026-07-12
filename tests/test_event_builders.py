"""S3-F1 A3.5: execute the REAL per-detector event builders (detector_events.py)
that the detectors themselves call -- no copied logic. Also pins A1 (first-wins
dedupe on collision) and A2 (torch frame = i + 1). Stdlib only.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cut_events import event_to_dict, validate_events  # noqa: E402
from detector_events import (  # noqa: E402
    build_minimal_events, build_mv_events, build_torch_events,
    build_transnet_events, gradual_span_frames,
)


class TestMinimal(unittest.TestCase):
    def test_sorted_minimal_and_deduped(self):
        evs = build_minimal_events([2.0, 1.0, 2.0])          # collision on 2.0
        self.assertEqual([e.time for e in evs], [1.0, 2.0])   # deduped + sorted
        self.assertEqual(event_to_dict(evs[0]), {"time": 1.0, "transition_kind": "unknown"})
        validate_events(evs)

    def test_empty(self):
        self.assertEqual(build_minimal_events([]), [])


class TestTorch(unittest.TestCase):
    def test_frame_is_i_plus_one(self):
        # A2: scores[i] is the change INTO frame i+1 -> frame = i + 1
        (e,) = build_torch_events([(296, 3.7)], fps=25.0)
        self.assertEqual(e.frame, 297)
        self.assertEqual(e.time, round(296 / 25.0, 4))        # time canonical, unchanged
        self.assertEqual(e.transition_kind, "unknown")
        self.assertEqual(e.confidence.metric, "hsv_spike_ratio")
        self.assertEqual(e.confidence.value, 3.7)             # native passthrough (>1 ok)
        validate_events([e])

    def test_collision_dedupe_first_wins(self):
        # two accepts whose times collide after rounding -> one event, first spike kept
        evs = build_torch_events([(1, 3.0), (2, 9.0)], fps=100000.0)   # both round to 0.0
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].time, 0.0)
        self.assertEqual(evs[0].frame, 2)                     # i=1 -> frame i+1 = 2 (first wins)
        self.assertEqual(evs[0].confidence.value, 3.0)


class TestMv(unittest.TestCase):
    def test_shape_and_pts_time(self):
        (e,) = build_mv_events([(12.3456, 296, 0.8)])
        self.assertEqual(e.frame, 296)                        # I-frame index (A2 conformant)
        self.assertEqual(e.time, round(12.3456, 3))           # 3-dp PTS
        self.assertEqual(e.confidence.metric, "mv_score")
        self.assertEqual(e.confidence.value, 0.8)
        self.assertEqual(e.transition_kind, "unknown")
        validate_events([e])

    def test_collision_dedupe_first_wins(self):
        evs = build_mv_events([(1.0001, 50, 0.8), (1.0002, 60, 0.9)])  # both round to 1.0
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].frame, 50)                    # first emitted wins
        self.assertEqual(evs[0].confidence.value, 0.8)


class TestGradualSpan(unittest.TestCase):
    def test_run_around_peak(self):
        prob = [0.0, 0.4, 0.55, 0.7, 0.5, 0.38, 0.2]           # >=0.35 run is frames 1..5
        self.assertEqual(gradual_span_frames(prob, 3, 0.35), (1, 5))

    def test_single_frame_run(self):
        self.assertEqual(gradual_span_frames([0.0, 0.9, 0.0], 1, 0.35), (1, 1))


class TestTransnet(unittest.TestCase):
    def test_hard_and_gradual_shapes(self):
        hard = [(24, 0.98)]
        gradual = [(60, 0.41, 58, 62)]
        evs, n_strobe = build_transnet_events(hard, gradual, 24.0, strobe_guard=False)
        self.assertEqual(n_strobe, 0)
        kinds = {e.transition_kind for e in evs}
        self.assertEqual(kinds, {"hard", "gradual"})
        hard_ev = next(e for e in evs if e.transition_kind == "hard")
        grad_ev = next(e for e in evs if e.transition_kind == "gradual")
        self.assertEqual(hard_ev.confidence.metric, "transnet_prob")
        self.assertIsNone(hard_ev.span)                       # never a span on a hard cut
        self.assertEqual(grad_ev.confidence.metric, "transnet_gradual_prob")
        self.assertEqual(grad_ev.span.start, round(58 / 24.0, 4))
        self.assertEqual(grad_ev.span.end, round(62 / 24.0, 4))
        validate_events(evs)

    def test_single_frame_gradual_omits_span(self):
        evs, _ = build_transnet_events([], [(50, 0.5, 50, 50)], 25.0, strobe_guard=False)
        self.assertIsNone(evs[0].span)

    def test_hard_wins_time_collision(self):
        # a hard cut and a gradual peak at the same rounded time -> hard kept (emitted first)
        evs, _ = build_transnet_events([(25, 0.9)], [(25, 0.5, 24, 26)], 25.0, strobe_guard=False)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].transition_kind, "hard")
        self.assertEqual(evs[0].confidence.metric, "transnet_prob")

    def test_strobe_suppression_runs(self):
        # 20 cuts at 0.5 s spacing -> pathologically dense -> dampen_strobe thins them
        hard = [(f * 3, 0.9) for f in range(20)]              # fps 6 -> times 0..9.5 by 0.5
        evs, n_strobe = build_transnet_events(
            hard, [], 6.0, strobe_guard=True,
            strobe_params=dict(dens_window=8, dens_max=8, merge_gap=4, keep_gap=2.5))
        self.assertGreater(n_strobe, 0)
        self.assertLess(len(evs), len(hard))
        validate_events(evs)


if __name__ == "__main__":
    unittest.main()
