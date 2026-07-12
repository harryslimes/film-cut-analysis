"""DetectResult is event-first (design §2 / scope item 2): events are the only
cut record and `cuts` is a read-only derived projection with no assignable float
list.

`detectors/base.py` is itself torch-free, but importing it as `detectors.base`
runs the package __init__ (which eagerly imports torch). So we load the module
file directly, bypassing the package -- keeping this test stdlib-only."""
import dataclasses
import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # so base.py's `from cut_events import ...` resolves

_spec = importlib.util.spec_from_file_location("cs_base_standalone",
                                               os.path.join(ROOT, "detectors", "base.py"))
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base   # dataclass annotation resolution looks the module up here
_spec.loader.exec_module(_base)
DetectResult = _base.DetectResult

from cut_events import CutEvent   # noqa: E402


def _result(times):
    return DetectResult(name="torch-gpu[cpu]", events=[CutEvent(time=t) for t in times],
                        elapsed=2.0, n_frames=100, fps_source=25.0)


class TestDetectResultEventFirst(unittest.TestCase):
    def test_cuts_is_projection_of_events(self):
        self.assertEqual(_result([1.0, 2.0, 3.0]).cuts, [1.0, 2.0, 3.0])

    def test_no_assignable_cuts_field(self):
        # `cuts` must be a derived property, not a dataclass field (§2).
        field_names = {f.name for f in dataclasses.fields(DetectResult)}
        self.assertIn("events", field_names)
        self.assertNotIn("cuts", field_names)

    def test_cuts_is_read_only(self):
        r = _result([1.0])
        with self.assertRaises(AttributeError):
            r.cuts = [9.9]                         # property has no setter

    def test_cuts_tracks_event_edits(self):
        r = _result([1.0, 2.0])
        r.events.append(CutEvent(time=3.0))
        self.assertEqual(r.cuts, [1.0, 2.0, 3.0])  # never a stale second copy

    def test_to_dict_carries_projection_not_events(self):
        d = _result([1.0, 2.0]).to_dict()
        self.assertEqual(d["cuts"], [1.0, 2.0])
        self.assertNotIn("events", d)
        self.assertNotIn("scores", d)


if __name__ == "__main__":
    unittest.main()
