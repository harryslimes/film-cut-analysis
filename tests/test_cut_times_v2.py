"""S3-E6: cut_times.py assembles the design-§3 v2 document (and batch_score tags
its cache). Torch-free: cut_times defers `from detectors import REGISTRY` into
main(), so importing the module and calling build_v2_document() needs no torch/
av/scenedetect -- only a real file for os.stat. The CLI end-to-end (a detector
run) is source-reviewed only; see the slice report's hand-test gate.
"""
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cut_times  # noqa: E402  (torch-free: REGISTRY import is deferred into main())
from cut_events import CutEvent, Confidence  # noqa: E402
from cut_export import validate_document, ExportValidationError  # noqa: E402

# a path that does not exist -> git provenance deterministically omitted (never faked)
NO_GIT = os.path.join(tempfile.gettempdir(), "cineshelf_s3e6_no_such_repo_dir")


def _res(events, *, name="transnetv2[cuda]", fps=25.0, n_frames=250, settings=None, extra=None):
    """Stub DetectResult: build_v2_document reads events/fps_source/n_frames/name/
    settings/extra."""
    return types.SimpleNamespace(events=events, fps_source=fps, n_frames=n_frames,
                                 name=name, settings=settings or {}, extra=extra or {})


def _events():
    return [CutEvent(time=1.0, frame=25, transition_kind="hard",
                     confidence=Confidence(0.98, "transnet_prob")),
            CutEvent(time=5.0, frame=125, transition_kind="hard",
                     confidence=Confidence(0.71, "transnet_prob"))]


class _TmpVideo:
    """A real temp file so os.stat succeeds; contents irrelevant."""
    def __enter__(self):
        fd, self.path = tempfile.mkstemp(suffix=".mkv")
        os.write(fd, b"x" * 2048); os.close(fd)
        return self.path

    def __exit__(self, *a):
        os.remove(self.path)


class TestBuildV2Document(unittest.TestCase):
    def _doc(self, detector_key="transnet-cuda", res=None):
        with _TmpVideo() as vid:
            r = res or _res(_events(), settings={"threshold": 0.4, "gradual_height": 0.35})
            return cut_times.build_v2_document(vid, detector_key, r, repo_dir=NO_GIT), vid

    def test_discriminators_and_legacy_keys(self):
        doc, vid = self._doc()
        self.assertEqual(doc["format"], "film-cut-analysis/cut-events")
        self.assertEqual(doc["schema_version"], 2)
        self.assertEqual((doc["fps"], doc["video"], doc["detector"]),
                         (25.0, vid, "transnetv2[cuda]"))

    def test_source_block(self):
        doc, vid = self._doc()
        s = doc["source"]
        self.assertEqual(s["path"], vid)
        self.assertEqual(s["size_bytes"], 2048)
        self.assertTrue(s["mtime_utc"].endswith("Z"))
        self.assertEqual(s["duration_seconds"], round(250 / 25.0, 3))   # 10.0

    def test_run_block_id_backend_model_settings(self):
        doc, _ = self._doc()
        run = doc["run"]
        self.assertEqual(run["detector_id"], "transnetv2")
        self.assertEqual(run["backend"], "cuda")
        self.assertEqual(run["model"], "transnetv2-pytorch (weights as installed)")
        self.assertEqual(run["settings"], {"threshold": 0.4, "gradual_height": 0.35})  # A3.1: from res.settings
        self.assertEqual(run["generated_by"], "cut_times.py")
        self.assertTrue(run["generated_utc"].endswith("Z"))

    def test_git_omitted_when_unavailable_never_faked(self):
        doc, _ = self._doc()
        self.assertNotIn("tool_commit", doc["run"])
        self.assertNotIn("tool_dirty", doc["run"])

    def test_analysis_block_and_derived_count(self):
        doc, _ = self._doc()
        a = doc["analysis"]
        self.assertEqual(a["status"], "complete")
        self.assertEqual(a["coverage"], [{"start": 0.0, "end": 10.0}])
        self.assertEqual(a["event_count"], 2)           # derived by the serializer
        # ffprobe can't time the fake temp file -> duration estimated from n_frames/fps,
        # which A3.4 records as a warning (naming the estimation).
        self.assertTrue(any("estimated" in w for w in a["warnings"]))

    def test_projection_holds_end_to_end(self):
        doc, _ = self._doc()
        self.assertEqual(doc["cuts"], [1.0, 5.0])
        self.assertEqual(doc["cuts"], [e["time"] for e in doc["cut_events"]])
        validate_document(doc)                          # full §3 revalidation passes

    def test_event_outside_coverage_is_rejected(self):
        # duration = 10.0; an event at 999s must fail the time-in-coverage check (§3.3)
        bad = [CutEvent(time=999.0, frame=1, transition_kind="hard")]
        with self.assertRaises(ExportValidationError):
            self._doc(res=_res(bad))


class TestMappingTable(unittest.TestCase):
    def test_id_backend_mapping(self):
        m = cut_times._DETECTOR_IDS
        self.assertEqual(m["transnet-cuda"], ("transnetv2", "cuda"))
        self.assertEqual(m["torch-cpu"], ("torch-gpu", "cpu"))
        self.assertEqual(m["ffmpeg-cuda"], ("ffmpeg-scene", "cuda"))
        self.assertEqual(m["psd-adaptive"], ("psd-adaptive", "cpu"))      # A3.6: truthful cpu
        self.assertEqual(m["motion-vectors"], ("motion-vectors", "cpu"))  # A3.6: truthful cpu

    def test_psd_backend_is_cpu(self):
        with _TmpVideo() as vid:
            doc = cut_times.build_v2_document(vid, "psd-adaptive",
                                              _res(_events(), name="psd-adaptive"), repo_dir=NO_GIT)
        self.assertEqual(doc["run"]["detector_id"], "psd-adaptive")
        self.assertEqual(doc["run"]["backend"], "cpu")  # A3.6: recorded, never omitted
        self.assertNotIn("model", doc["run"])           # non-transnet -> no model

    def test_unknown_key_falls_back_to_raw(self):
        self.assertEqual(cut_times._DETECTOR_IDS.get("future-x", ("future-x", None)),
                         ("future-x", None))


class TestDurationAndWarnings(unittest.TestCase):
    def test_missing_duration_omits_coverage_and_warns(self):
        # n_frames = 0 -> duration 0 -> coverage omitted; §3.3 relaxes to time >= 0
        with _TmpVideo() as vid:
            doc = cut_times.build_v2_document(
                vid, "torch-cpu", _res(_events(), name="torch-gpu[cpu]", n_frames=0),
                repo_dir=NO_GIT)
        self.assertNotIn("duration_seconds", doc["source"])
        self.assertEqual(doc["analysis"]["coverage"], [])
        self.assertTrue(any("duration unavailable" in w for w in doc["analysis"]["warnings"]))
        validate_document(doc)                          # still valid (no coverage bound)

    def test_motion_vectors_warning_surfaced(self):
        extra = {"keyint": 5.0, "fixed_gop": True,
                 "warning": "fixed-GOP encode: I-frames are periodic, not scene-cuts"}
        with _TmpVideo() as vid:
            doc = cut_times.build_v2_document(
                vid, "motion-vectors",
                _res([CutEvent(time=2.0, frame=48, confidence=Confidence(0.5, "mv_score"))],
                     name="motion-vectors", extra=extra),
                repo_dir=NO_GIT)
        self.assertIn(extra["warning"], doc["analysis"]["warnings"])
        self.assertEqual(doc["run"]["backend"], "cpu")   # A3.6


class TestGitProvenanceHelper(unittest.TestCase):
    def test_bad_dir_omits_both(self):
        commit, dirty = cut_times._git_provenance(NO_GIT)
        self.assertIsNone(commit)
        self.assertIsNone(dirty)

    def test_real_repo_returns_hex_commit_when_git_present(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        commit, dirty = cut_times._git_provenance(repo)
        if commit is not None:                          # tolerant if git is absent
            self.assertEqual(len(commit), 40)
            self.assertTrue(all(c in "0123456789abcdef" for c in commit))
            self.assertIsInstance(dirty, bool)


class TestLegacyReader(unittest.TestCase):
    def test_v2_doc_readable_by_a_legacy_reader(self):
        # a legacy consumer wants only video / fps / cuts -- all still present at top level
        with _TmpVideo() as vid:
            doc = cut_times.build_v2_document(vid, "transnet-cuda", _res(_events()), repo_dir=NO_GIT)
        self.assertEqual(doc["video"], vid)
        self.assertEqual(doc["fps"], 25.0)
        self.assertEqual(doc["cuts"], [1.0, 5.0])


class TestFpsGuard(unittest.TestCase):
    """A3.4: non-positive / non-finite fps is a hard error before export, never faked."""
    def _build(self, fps):
        with _TmpVideo() as vid:
            return cut_times.build_v2_document(vid, "torch-cpu",
                                               _res(_events(), name="torch-gpu[cpu]", fps=fps),
                                               repo_dir=NO_GIT)

    def test_zero_fps_rejected(self):
        self.assertRaises(ExportValidationError, self._build, 0.0)

    def test_negative_fps_rejected(self):
        self.assertRaises(ExportValidationError, self._build, -5.0)

    def test_nan_fps_rejected(self):
        self.assertRaises(ExportValidationError, self._build, float("nan"))


class TestDecodeStatus(unittest.TestCase):
    """A3.4: a detector that owns its decode reports decode_ok; export downgrades status."""
    def test_partial_when_decode_failed_but_events_exist(self):
        with _TmpVideo() as vid:
            doc = cut_times.build_v2_document(
                vid, "torch-cpu",
                _res(_events(), name="torch-gpu[cpu]",
                     extra={"decode_ok": False, "decode_detail": "ffmpeg exited 1"}),
                repo_dir=NO_GIT)
        self.assertEqual(doc["analysis"]["status"], "partial")
        self.assertIn("ffmpeg exited 1", doc["analysis"]["warnings"])

    def test_failed_when_decode_failed_and_no_events(self):
        with _TmpVideo() as vid:
            doc = cut_times.build_v2_document(
                vid, "torch-cpu",
                _res([], name="torch-gpu[cpu]",
                     extra={"decode_ok": False, "decode_detail": "ffmpeg exited 1"}),
                repo_dir=NO_GIT)
        self.assertEqual(doc["analysis"]["status"], "failed")


class TestNonJsonWritersUnchanged(unittest.TestCase):
    """CSV / EDL / SRT writers untouched -- exercise them (stdlib-only, no detector)."""
    def _tmp(self):
        fd, p = tempfile.mkstemp(); os.close(fd); return p

    def test_csv(self):
        p = self._tmp()
        try:
            cut_times.write_csv(p, [1.0, 2.5], 25.0)
            with open(p) as fh:
                lines = fh.read().splitlines()
            self.assertEqual(lines[0], "cut_index,time_seconds,timecode")
            self.assertEqual(lines[1], "1,1.000,00:00:01:00")
        finally:
            os.remove(p)

    def test_srt_and_edl_smoke(self):
        for writer in ("srt", "edl"):
            p = self._tmp()
            try:
                if writer == "srt":
                    cut_times.write_srt(p, [1.0, 2.0], 3.0)
                else:
                    cut_times.write_edl(p, [1.0, 2.0], 25.0, 3.0)
                self.assertTrue(os.path.getsize(p) > 0)
            finally:
                os.remove(p)


if __name__ == "__main__":
    unittest.main()
