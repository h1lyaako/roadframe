import io
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

from analysis import decoded_frames, video_info
from server import app
from video_frames import samples


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.folder = Path(cls.temp.name)
        cls.client = app.test_client()
        cls.odd = cls.folder / "odd.mkv"
        cls.tail = cls.folder / "tail.mp4"
        base = ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i"]
        subprocess.run(base + ["testsrc=size=161x91:rate=12:duration=3",
                               "-c:v", "ffv1", str(cls.odd)], check=True)
        subprocess.run(base + ["color=size=162x92:rate=25:duration=2",
                               "-f", "lavfi", "-i", "anullsrc=sample_rate=48000",
                               "-t", "6", "-c:v", "libx264", "-c:a", "aac", str(cls.tail)], check=True)

    def upload(self, data, name):
        res = self.client.post("/api/jobs", data={"video": (io.BytesIO(data), name)})
        self.assertEqual(res.status_code, 202)
        url = "/api/jobs/" + res.json["id"]
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            job = self.client.get(url).json
            if job["status"] != "working":
                return url, job
            time.sleep(.05)
        self.fail(f"Processing did not finish: {name}")

    def read_video(self, url, name):
        with self.client.get(url) as res:
            self.assertEqual(res.status_code, 200)
            out = self.folder / name
            out.write_bytes(res.data)
        return video_info(out)

    def test_video_duration_and_export_bounds(self):
        url, job = self.upload(self.tail.read_bytes(), self.tail.name)
        self.assertEqual(job["status"], "ready", job)
        self.assertAlmostEqual(job["duration"], 2, places=2)
        preview = self.read_video(url + "/video", "tail-preview.mp4")
        self.assertTrue(preview.browser_ready)
        self.assertAlmostEqual(preview.duration, 2, places=2)
        for query in ("start=1&end=5", "start=-1&end=1", "start=0&end=nan", "end=1"):
            with self.subTest(query=query):
                self.assertEqual(self.client.get(url + "/clip?" + query).status_code, 400)

    def test_odd_dimensions_preview_and_clip(self):
        url, job = self.upload(self.odd.read_bytes(), self.odd.name)
        self.assertEqual(job["status"], "ready", job)
        self.assertTrue(self.read_video(url + "/video", "odd-preview.mp4").browser_ready)
        info = self.read_video(url + "/clip?start=.5&end=2", "odd-clip.mp4")
        self.assertTrue(info.browser_ready)
        self.assertAlmostEqual(info.duration, 1.5, delta=1/12)

    def test_corrupt_upload_releases_queue(self):
        url, bad = self.upload(b"broken video", "broken.mp4")
        self.assertEqual(bad["status"], "error")
        self.assertEqual(self.client.get(url + "/video").status_code, 404)
        _, good = self.upload(self.tail.read_bytes(), self.tail.name)
        self.assertEqual(good["status"], "ready", good)

    def test_cloud_mode_requires_explicit_selection(self):
        self.assertEqual(self.client.post("/api/jobs?mode=codex").status_code, 400)

    def test_recording_without_duration_metadata(self):
        stream = self.folder / "stream.mkv"
        stream.write_bytes(subprocess.check_output([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(self.odd), "-map_metadata", "-1",
            "-c", "copy", "-f", "matroska", "pipe:1",
        ]))
        self.assertAlmostEqual(video_info(stream).duration, 3, delta=.01)

    def test_all_detectors_and_export_use_the_same_camera(self):
        dual = self.folder / "two-views.mkv"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        "color=black:size=160x90:rate=10:duration=2", "-f", "lavfi", "-i",
                        "color=white:size=320x180:rate=10:duration=4", "-map", "0:v", "-map", "1:v",
                        "-c:v", "ffv1", "-disposition:v:0", "0", "-disposition:v:1", "default",
                        str(dual)], check=True)
        frames = list(decoded_frames(dual, 5))
        self.assertEqual(len(frames), 10)
        self.assertLess(frames[0].mean(), 5)
        self.assertLess(samples(dual, 0, 1, 1, 32, 18)[0][1].mean(), 5)
        url, job = self.upload(dual.read_bytes(), dual.name)
        self.assertEqual(job["status"], "ready", job)
        self.assertAlmostEqual(job["duration"], 2, places=2)
        self.read_video(url + "/clip?start=0&end=1", "first-camera.mp4")
        self.assertLess(samples(self.folder / "first-camera.mp4", 0, 1, 1, 32, 18)[0][1].mean(), 5)
