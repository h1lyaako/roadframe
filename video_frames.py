"""Sampling and spatial crops shared by the collision detectors."""

import subprocess

import numpy as np


def samples(path, start, span, fps, width, height):
    """Decode sequentially within a window. Labels indicate sample times, not camera clock."""
    # round=up chooses the source frame at or before the requested instant.
    vf = (
        f"fps={fps}:round=up:start_time=0,scale={width}:{height}:"
        f"force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-threads",
        "2",
        "-i",
        str(path),
        "-t",
        f"{span:.3f}",
        "-map",
        "0:V:0",
        "-an",
        "-sn",
        "-vf",
        vf,
        "-pix_fmt",
        "bgr24",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    data = subprocess.check_output(cmd, timeout=120, stderr=subprocess.DEVNULL)
    images = np.frombuffer(data, np.uint8).reshape(-1, height, width, 3)
    return [(round(start + i / fps, 3), img) for i, img in enumerate(images) if i / fps < span]


def crop_region(frames, region):
    x, y, size = region
    left = min(1 - size, max(0, x - size / 2))
    top = min(1 - size, max(0, y - size / 2))
    h, w = frames[0][1].shape[:2]
    return [
        (t, im[int(top * h) : int((top + size) * h), int(left * w) : int((left + size) * w)])
        for t, im in frames
    ]
