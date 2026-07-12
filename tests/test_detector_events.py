"""S3-E2: every detector constructs the event-first DetectResult with MINIMAL
events -- wrap_times(cuts), all optional fields omitted, transition_kind left at
its "unknown" default (design §2, §3.2). No confidence/frame/kind yet (S3-E3/4/5).

Torch-free verification. The detector modules cannot be imported here: the
`detectors` package __init__ eagerly imports torch (and the rich detectors need
numpy/scenedetect/av), none of which we may install; and actually *running* a
detector needs a real video, which is out of scope. So we verify two things
without those deps:
  1. the construction PATTERN each detector now uses --
     `DetectResult(events=wrap_times([...]), ...)` -- yields valid minimal events;
  2. by source scan, that all five constructors were converted off the old
     `cuts=` float kwarg.
"""
import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # so base.py's `from cut_events import ...` resolves

from cut_events import event_to_dict, validate_events, wrap_times   # noqa: E402

# Load detectors/base.py directly (torch-free): importing it as `detectors.base`
# would run the package __init__ and pull torch. Register in sys.modules before
# exec so Py3.14 dataclass annotation resolution can find the module.
_spec = importlib.util.spec_from_file_location(
    "cs_base_standalone_e2", os.path.join(ROOT, "detectors", "base.py"))
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)
DetectResult = _base.DetectResult

# every DetectResult(...) constructor in the repo's detector family
DETECTOR_FILES = [
    "detectors/ffmpeg_scene.py",
    "detectors/pyscenedetect.py",
    "detectors/torch_gpu.py",
    "detectors/transnet.py",
    "detectors/motion_vectors.py",
]


class TestConstructionPattern(unittest.TestCase):
    """The exact path each detector now takes: floats -> wrap_times -> DetectResult."""

    def test_minimal_events_are_valid_and_omit_optionals(self):
        cuts = [1.0, 2.5, 9.0]          # the sorted float list every detector holds
        r = DetectResult(name="detector", events=wrap_times(cuts),
                         elapsed=1.0, n_frames=100, fps_source=25.0)
        self.assertEqual(r.cuts, cuts)                  # legacy projection round-trips
        validate_events(r.events)                       # §3.3 event rules pass
        for e in r.events:
            self.assertEqual(e.transition_kind, "unknown")
            self.assertEqual(event_to_dict(e), {"time": e.time, "transition_kind": "unknown"})

    def test_base_re_exports_wrap_times(self):
        # detectors import it via `from .base import ... wrap_times`
        self.assertIs(_base.wrap_times, wrap_times)

    def test_empty_detection_is_valid(self):
        r = DetectResult(name="detector", events=wrap_times([]),
                         elapsed=1.0, n_frames=1, fps_source=25.0)
        self.assertEqual(r.cuts, [])
        validate_events(r.events)


class TestAllDetectorsConverted(unittest.TestCase):
    """Source scan -- covers the detectors we cannot import without heavy deps."""

    def _src(self, rel):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            return fh.read()

    def test_every_detector_constructs_with_events(self):
        # Durable cross-detector invariant: each builds DetectResult with an `events=`
        # arg (minimal detectors via wrap_times(...); enriched ones, e.g. torch_gpu from
        # S3-E3, build CutEvents directly) -- never the old float container.
        for rel in DETECTOR_FILES:
            src = self._src(rel)
            self.assertIn("DetectResult(", src, f"{rel}: no DetectResult construction")
            self.assertIn("events=", src, f"{rel}: not constructing with events=")

    def test_no_detector_still_passes_a_cuts_kwarg(self):
        # the old kwarg was written `cuts=cuts` (no space); local assignments use
        # `cuts = ...` (with space), so a bare `cuts=` substring flags only a kwarg.
        for rel in DETECTOR_FILES:
            self.assertNotIn("cuts=", self._src(rel), f"{rel}: still passes a cuts= kwarg")


if __name__ == "__main__":
    unittest.main()
