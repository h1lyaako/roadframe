from pathlib import Path
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from archive_scan import packet_overview, visual_overview, window_starts
from night_motion import lost_light_events


class SearchTests(unittest.TestCase):
    def test_light_track_needs_a_disappearance_and_nearby_braking(self):
        def sequence(braking):
            frames = []
            for i in range(30):
                frame = np.zeros((360, 640, 3), np.uint8)
                if i < 10:
                    x = [180, 194, 209, 224, 237, 247, 253, 257, 259, 260][i]
                    cv2.circle(frame, (x, 188), 3, (0, 0, 220), -1)
                if braking and 17 <= i <= 23:
                    cv2.circle(frame, (290, 180), 11, (0, 0, 220), -1)
                frames.append((i / 6, frame))
            return frames
        found = lost_light_events(sequence(True))
        self.assertEqual([e["status"] for e in found], ["light_track"])
        self.assertAlmostEqual(found[0]["time"], 10 / 6)
        self.assertEqual(lost_light_events(sequence(False)), [])

    def test_late_activity_is_eligible_for_detail(self):
        scores = np.zeros(3600)
        scores[3586] = 10
        starts = window_starts(scores, np.array([]), None, 3600, np.array([1805]))
        self.assertTrue(any(t <= 3586 < t + 20 for t in starts))
        self.assertTrue(any(t <= 1805 < t + 20 for t in starts))
        self.assertTrue(all(0 <= t <= 3580 for t in starts))

    def test_sampled_keyframes_stay_aligned(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "intraframes.mkv"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=64x48:rate=48:duration=44", "-c:v", "libx264",
                            "-threads", "2", "-preset", "ultrafast", "-g", "1", str(path)], check=True)
            _, keys, _ = packet_overview(path, 44)
            values, times = visual_overview(path, keys)
            self.assertEqual(len(values), len(keys[::2]))
            np.testing.assert_array_equal(times, keys[::2])
