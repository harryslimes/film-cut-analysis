"""Amendment 2 A2-2: transnet.detect reports a failed/partial decode through the
decode_ok/decode_detail channel (A3.4 parity) instead of crashing on a nonzero ffmpeg
exit (it used check=True). Both branches are exercised with the ffmpeg subprocess
mocked: the failure branch must never touch the model, and the model is mocked for the
clean-exit branch so the test stays light. The detector imports torch+numpy at module
load, so this file is skipped where those are unavailable.
"""
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy
    import torch
    from detectors import transnet
    HAVE_DEPS = True
except Exception:                                    # torch/numpy not installed
    HAVE_DEPS = False

FRAME_BYTES = 27 * 48 * 3


def _proc(returncode, stdout=b"", stderr=b""):
    """A stand-in for subprocess.run's CompletedProcess."""
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


@unittest.skipUnless(HAVE_DEPS, "transnet detector needs torch+numpy")
class TestTransnetGracefulDecode(unittest.TestCase):
    def test_nonzero_exit_empty_output_is_reported_not_raised(self):
        # the real 10-bit blocker shape: hwdownload EINVAL, nothing on stdout.
        stderr = b"[hwdownload] Invalid output format nv12 for hwframe download.\n"
        with mock.patch.object(transnet, "ffprobe_info", return_value=(24.0, 100, 1920, 1080)), \
             mock.patch.object(transnet.subprocess, "run",
                               return_value=_proc(4294967274, b"", stderr)) as run, \
             mock.patch.object(transnet, "_model") as model:
            res = transnet.detect("dummy.mkv", method="cuda")   # must NOT raise
        run.assert_called_once()
        model.assert_not_called()                    # never spun up the GPU model
        self.assertEqual(res.events, [])
        self.assertEqual(res.n_frames, 0)
        self.assertFalse(res.extra["decode_ok"])
        self.assertIn("Invalid output format nv12", res.extra["decode_detail"])
        self.assertEqual(res.settings["method"], "cuda")   # resolved settings still present

    def test_clean_exit_with_frames_completes(self):
        n = 8
        stdout = bytes(n * FRAME_BYTES)              # n black frames worth of rgb24 bytes
        fake = mock.Mock()
        fake.device = "cpu"
        fake.predict_frames.return_value = (torch.zeros(n), torch.zeros(n))
        fake.predictions_to_scenes.return_value = numpy.array([[0, 3], [4, n - 1]])
        with mock.patch.object(transnet, "ffprobe_info", return_value=(24.0, n, 96, 54)), \
             mock.patch.object(transnet.subprocess, "run", return_value=_proc(0, stdout, b"")), \
             mock.patch.object(transnet, "_model", return_value=fake):
            res = transnet.detect("dummy.mkv", method="cuda")
        self.assertNotIn("decode_ok", res.extra)     # clean exit -> exporter status 'complete'
        self.assertEqual(res.n_frames, n)
        self.assertGreaterEqual(len(res.events), 1)
        fake.predict_frames.assert_called_once()


if __name__ == "__main__":
    unittest.main()
