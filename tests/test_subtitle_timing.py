"""End-to-end test that the clip caption is burned onto the EXACT frame it should be.

This is a real render test, not a mock: it stands up the actual `label_server`, asks it for
a real transcoded clip (`/clip`) and its WebVTT caption track (`/clipvtt`), burns the caption
onto the clip with ffmpeg's libass `subtitles` filter (the same renderer players use for the
cue timing), then decodes every output frame and checks, pixel-by-pixel, that the caption text
appears on the frame at the cut and on none before it.

The contract under test: with `only=1`, the single "Cut #N" caption must appear at
clip-time == `pre` (exactly the cut) and hold through the tail to the clip end.

Requires: ffmpeg (with the libass `subtitles` filter), Pillow, numpy. If any are missing the
test SKIPs rather than failing, so it never gives a fake green.

Run:  python3 tests/test_subtitle_timing.py       (or via pytest / unittest discover)
"""
import os
import re
import shutil
import subprocess
import sys
import time
import unittest
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8023
FPS = 30
CUT_T = 4.0          # cut sits at 4.0s in the (synthetic) source
PRE, POST = 2.0, 0.5  # clip = [2.0, 4.5]; cut is at clip-time == PRE == 2.0s
# cuts.json so the caption's ordinal is real: the cut at 4.0 is the 3rd cut → "Cut #3"
SOURCE_CUTS = [1.0, 2.5, 4.0]
EXPECT_ORDINAL = 3
WHITE_MIN = 180      # a pixel is "caption ink" if every channel is at least this bright
WHITE_PIXELS = 15    # a frame "has a caption" if at least this many caption-ink pixels appear


def _have(cmd):
    return shutil.which(cmd) is not None


def _ffmpeg_has_subtitles_filter():
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                             capture_output=True, text=True).stdout
        return bool(re.search(r"^\s*\S*\s+subtitles\s", out, re.M))
    except OSError:
        return False


def _deps_ok():
    if not (_have("ffmpeg") and _have("ffprobe") and _ffmpeg_has_subtitles_filter()):
        return "ffmpeg/ffprobe with the subtitles filter is required"
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return "numpy and Pillow are required"
    return None


class SubtitleFrameTimingTest(unittest.TestCase):
    server = None
    tmp = None

    @classmethod
    def setUpClass(cls):
        reason = _deps_ok()
        if reason:
            raise unittest.SkipTest(reason)
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix="subtitle_test_")
        cls.src = os.path.join(cls.tmp, "synthetic.mp4")
        # A black source so the ONLY bright pixels in the frame come from the white caption ink.
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", f"color=c=black:s=640x360:r={FPS}", "-t", "6",
             "-pix_fmt", "yuv420p", "-c:v", "libx264", cls.src],
            check=True)
        # Give it a real base run so /clipvtt computes the true cut ordinal ("Cut #3").
        import json
        with open(cls.src.rsplit(".", 1)[0] + ".cuts.json", "w") as fh:
            json.dump({"cuts": SOURCE_CUTS}, fh)
        # Boot the actual server.
        cls.server = subprocess.Popen(
            [sys.executable, "label_server.py", "--port", str(PORT)],
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls._wait_ready()

    @classmethod
    def _wait_ready(cls):
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/", timeout=1).read()
                return
            except OSError:
                time.sleep(0.25)
        raise RuntimeError("label_server did not come up on port %d" % PORT)

    @classmethod
    def tearDownClass(cls):
        if cls.server:
            cls.server.terminate()
            try:
                cls.server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server.kill()
        if cls.tmp and os.path.isdir(cls.tmp):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def _get(self, path):
        with urllib.request.urlopen(f"http://localhost:{PORT}{path}", timeout=60) as r:
            return r.read()

    def _qs(self, **kw):
        return urllib.parse.urlencode(kw)

    # ---- 1. the caption track itself is timed to the cut -----------------------------------
    def test_vtt_cue_is_timed_exactly_at_the_cut(self):
        vtt = self._get("/clipvtt?" + self._qs(
            v=self.src, t=CUT_T, pre=PRE, post=POST, mark=CUT_T, only=1)).decode()
        # exactly one cue
        cues = re.findall(r"(\d\d:\d\d:\d\d\.\d\d\d) --> (\d\d:\d\d:\d\d\.\d\d\d)\n(.+)", vtt)
        self.assertEqual(len(cues), 1, f"expected one caption, got: {vtt!r}")
        start, end, text = cues[0]

        def secs(ts):
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)

        # appears AT the cut (clip-time == PRE) and holds to the clip end (PRE+POST)
        self.assertAlmostEqual(secs(start), PRE, places=3)
        self.assertAlmostEqual(secs(end), PRE + POST, places=3)
        self.assertEqual(text.strip(), f"Cut #{EXPECT_ORDINAL}")

    # ---- 2. rendered pixels: the caption is on the cut frame and none before ---------------
    def test_caption_burns_onto_the_exact_cut_frame(self):
        import numpy as np
        from PIL import Image

        clip = os.path.join(self.tmp, "clip.mp4")
        vtt = os.path.join(self.tmp, "sub.vtt")
        with open(clip, "wb") as fh:
            fh.write(self._get("/clip?" + self._qs(v=self.src, t=CUT_T, pre=PRE, post=POST)))
        with open(vtt, "wb") as fh:
            fh.write(self._get("/clipvtt?" + self._qs(
                v=self.src, t=CUT_T, pre=PRE, post=POST, mark=CUT_T, only=1)))
        self.assertGreater(os.path.getsize(clip), 0, "server returned an empty clip")

        # Burn the caption in exactly as a player would render the cue, then dump every frame.
        # Run from tmp so the subtitles-filter arg has no path separators to escape.
        frames_dir = os.path.join(self.tmp, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-i", "clip.mp4",
             "-vf", "subtitles=sub.vtt,showinfo", "-start_number", "0",
             os.path.join("frames", "f_%05d.png")],
            cwd=self.tmp, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"burn failed:\n{proc.stderr[-2000:]}")

        # showinfo prints one line per frame, in order, with the frame index and its pts_time.
        info = re.findall(r"\bn:\s*(\d+)\b.*?\bpts_time:\s*([0-9.]+)", proc.stderr)
        self.assertTrue(info, "no showinfo frame data parsed")
        pts_by_n = {int(n): float(t) for n, t in info}
        t0 = min(pts_by_n.values())  # normalise: clip should start at 0

        def caption_ink(path):
            im = np.asarray(Image.open(path).convert("RGB"))
            band = im[int(im.shape[0] * 0.70):, :, :]  # captions render in the bottom band
            return int(np.count_nonzero(np.all(band >= WHITE_MIN, axis=2)))

        frames = []  # (clip_time, ink_pixels)
        for fn in sorted(os.listdir(frames_dir)):
            m = re.match(r"f_(\d+)\.png", fn)
            if not m:
                continue
            n = int(m.group(1))
            if n not in pts_by_n:
                continue
            frames.append((pts_by_n[n] - t0, caption_ink(os.path.join(frames_dir, fn))))
        frames.sort()
        self.assertGreater(len(frames), 30, "too few frames decoded")

        frame_dur = 1.0 / FPS
        tol = 1.5 * frame_dur  # allow one frame of slack for encode rounding

        # onset = first frame carrying a caption
        onset = next((t for t, ink in frames if ink >= WHITE_PIXELS), None)
        self.assertIsNotNone(onset, "caption never appeared on any frame")

        # (a) frame-exact: the caption turns on at the cut (clip-time == PRE), within one frame
        self.assertLessEqual(
            abs(onset - PRE), tol,
            f"caption onset {onset:.3f}s is not within a frame of the cut at {PRE:.3f}s")

        # (b) nothing before the cut carries a caption
        for t, ink in frames:
            if t <= PRE - tol:
                self.assertLess(
                    ink, WHITE_PIXELS,
                    f"caption ink ({ink}px) appeared BEFORE the cut, at clip-time {t:.3f}s")

        # (c) the caption holds from the cut through the tail (to the clip end)
        held = [(t, ink) for t, ink in frames if PRE + frame_dur <= t <= PRE + POST - frame_dur]
        self.assertTrue(held, "no frames in the post-cut hold window")
        for t, ink in held:
            self.assertGreaterEqual(
                ink, WHITE_PIXELS,
                f"caption vanished during its hold window at clip-time {t:.3f}s (ink {ink}px)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
