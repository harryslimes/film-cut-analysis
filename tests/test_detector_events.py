"""S3-F1: source-scan invariants across all five detectors. Each constructs the
event-first DetectResult by calling a dependency-light builder (detector_events.py),
never the old float `cuts=` kwarg, and populates the resolved `settings=` field
(A3.1, diagnostics kept separate in `extra`). Dependency-light: reads source only.

The event-construction logic itself is executed (not scanned) by test_event_builders.py.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DETECTOR_FILES = [
    "detectors/ffmpeg_scene.py",
    "detectors/pyscenedetect.py",
    "detectors/torch_gpu.py",
    "detectors/transnet.py",
    "detectors/motion_vectors.py",
]


class TestAllDetectorsConverted(unittest.TestCase):
    def _src(self, rel):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            return fh.read()

    def test_every_detector_builds_events_via_a_builder(self):
        for rel in DETECTOR_FILES:
            src = self._src(rel)
            self.assertIn("DetectResult(", src, f"{rel}: no DetectResult construction")
            self.assertIn("events=", src, f"{rel}: not constructing with events=")
            self.assertRegex(src, r"build_(minimal|torch|mv|transnet)_events\(")

    def test_no_detector_passes_a_cuts_kwarg(self):
        # the old kwarg was `cuts=cuts` (no space); local assignments use `cuts = ...`
        # (with space), so a bare `cuts=` substring flags only a kwarg.
        for rel in DETECTOR_FILES:
            self.assertNotIn("cuts=", self._src(rel), f"{rel}: still passes a cuts= kwarg")

    def test_every_detector_populates_resolved_settings(self):
        for rel in DETECTOR_FILES:
            self.assertIn("settings=", self._src(rel), f"{rel}: no resolved settings= (A3.1)")


if __name__ == "__main__":
    unittest.main()
