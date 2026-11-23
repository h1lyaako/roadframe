from pathlib import Path
import subprocess
import tempfile
import unittest

from analysis import analyze, probe


ROOT = Path(__file__).resolve().parents[1]
RU = ROOT / "test-videos-ru"
OTHER = ROOT / "test-videos"


class RecordingTests(unittest.TestCase):
    def check_marks(self, path, windows):
        duration, _ = probe(path)
        times = [e["time"] for e in analyze(path, duration)]
        self.assertEqual(len(times), len(windows), f"{path.name}: {times}")
        for timepoint, (lo, hi) in zip(times, windows):
            self.assertTrue(lo <= timepoint <= hi, f"{path.name}: {timepoint} outside {lo}–{hi}")

    def test_known_collisions(self):
        cases = [
            (RU / "01-trekhgorny.webm", (15.0, 17.0)),
            (RU / "03-chelyabinsk.mp4", (8.0, 10.0)),
            (RU / "04-saint-petersburg.mp4", (3.5, 5.25)),
            (OTHER / "01-lane-change.webm", (2.5, 4.5)),
            (OTHER / "05-teslacam-crash.webm", (15, 17)),
        ]
        for path, window in cases:
            with self.subTest(path=path.name):
                self.check_marks(path, [window])

    def test_no_new_impact(self):
        for path in (RU / "07-trekhgorny-control.mp4", OTHER / "02-highway-pileup.webm",
                     OTHER / "03-highway-400.webm", OTHER / "04-normal-drive.mp4"):
            with self.subTest(path=path.name):
                self.check_marks(path, [])

    def test_ordinary_cross_traffic(self):
        # Use only the verified first seconds. The rest of this recording is ambiguous.
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / "crossing.mp4"
            self.make_clip(RU / "02-ozersk.webm", clip, 0, 5)
            self.check_marks(clip, [])

    def test_shifted_roadside_impact(self):
        # Different time origin, frame size and encoding. Traffic after the crash
        # must not create a second mark as other vehicles approach the junction.
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / "junction.mp4"
            self.make_clip(RU / "01-trekhgorny.webm", clip, 6, 28)
            self.check_marks(clip, [(9, 11)])

    @staticmethod
    def make_clip(source, target, start, duration):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", str(source),
                        "-t", str(duration), "-an", "-vf", "scale=854:-2", "-c:v", "libx264",
                        "-threads", "2", "-preset", "veryfast", str(target)], check=True)
