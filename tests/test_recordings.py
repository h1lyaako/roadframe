from pathlib import Path
import subprocess
import tempfile
import unittest

from analysis import analyze, probe
from impact_audio import impact_sounds


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

    def test_distant_and_obscured_episodes(self):
        for name, lo, hi in (("05-moscow-night.mp4", 8.5, 9.5), ("06-moscow-day.mp4", 7.4, 8.5)):
            with self.subTest(path=name):
                path = RU / name
                events = analyze(path, probe(path)[0])
                self.assertEqual(len(events), 1, events)
                self.assertTrue(lo <= events[0]["time"] <= hi, events)
                self.assertEqual(events[0]["status"], "audiovisual_candidate")

    def test_sound_without_motion(self):
        source = RU / "06-moscow-day.mp4"
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / "static.mp4"
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=s=256x144:r=10:d=14",
                "-i", str(source), "-map", "0:v", "-map", "1:a:0", "-c:v", "libx264",
                "-threads", "2", "-c:a", "copy", "-shortest", str(clip),
            ], check=True)
            sounds = impact_sounds(clip, probe(clip)[0])
            self.assertTrue(any(8 < t < 9 for t in sounds), sounds)
            self.check_marks(clip, [])

    def test_moved_audio_and_video(self):
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / "night.mp4"
            self.make_clip(RU / "05-moscow-night.mp4", clip, 3, 12, audio=True)
            self.check_marks(clip, [(5.4, 6.5)])

    def test_delayed_audio_track(self):
        source = RU / "06-moscow-day.mp4"
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / "delayed.mp4"
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-itsoffset", "2",
                "-i", str(source), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(clip),
            ], check=True)
            duration = probe(clip)[0]
            original = impact_sounds(source, duration)
            delayed = impact_sounds(clip, duration)
            self.assertEqual(len(original), 1)
            self.assertEqual(len(delayed), 1)
            self.assertAlmostEqual(delayed[0] - original[0], 2.0, delta=.09)

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
    def make_clip(source, target, start, duration, audio=False):
        audio_args = ["-c:a", "aac"] if audio else ["-an"]
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", str(source),
                        "-t", str(duration), *audio_args, "-vf", "scale=854:-2", "-c:v", "libx264",
                        "-threads", "2", "-preset", "veryfast", str(target)], check=True)
