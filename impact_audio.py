"""Broadband transients on the first audio track, relative to the first video frame."""
import json
import subprocess

import numpy as np


RATE = 16000
FRAME = 512
WINDOW = np.hanning(FRAME)
FREQUENCIES = np.fft.rfftfreq(FRAME, 1 / RATE)
BANDS = [(FREQUENCIES >= lo) & (FREQUENCIES < hi)
         for lo, hi in ((300, 1000), (1000, 3000), (3000, 7500))]


def impact_sounds(path, duration):
    try:
        tracks = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time:stream_disposition=attached_pic",
            "-of", "json", str(path),
        ], capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        return []
    if tracks.returncode:
        return []
    streams = json.loads(tracks.stdout).get("streams", [])
    if not any(s["codec_type"] == "audio" for s in streams):
        return []
    video = next((s for s in streams if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        return []
    origin = video.get("start_time")
    start = float(origin) if origin not in (None, "N/A") else 0
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-copyts", "-i", str(path), "-map", "0:a:0",
        "-af", f"asetpts=PTS-({start})/TB,aresample={RATE}:async=1:first_pts=0",
        "-ac", "1", "-t", str(duration), "-f", "f32le", "pipe:1",
    ]
    features = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as decoder:
        while raw := decoder.stdout.read(256 * FRAME * 4):
            count = len(raw) // (FRAME * 4)
            if not count:
                break
            pcm = np.frombuffer(raw, "<f4", count=count * FRAME).reshape(count, FRAME)
            power = np.abs(np.fft.rfft(pcm * WINDOW)) ** 2
            levels = [10 * np.log10(power[:, band].mean(axis=1) + 1e-10) for band in BANDS]
            high = power[:, FREQUENCIES >= 1000]
            flatness = np.exp(np.log(high + 1e-12).mean(axis=1)) / (high.mean(axis=1) + 1e-12)
            features.append(np.column_stack([*levels, flatness]).astype(np.float32))
        if decoder.wait() or not features:
            return []
    return transient_times(np.concatenate(features))


def transient_times(features):
    levels = features[:, :3]
    hits = []
    # Require a broadband rise and decay relative to recent road noise.
    for i in np.flatnonzero((features[:, 3] >= .07) & (levels[:, 2] > -36)):
        if i < 22 or i + 12 > len(levels):
            continue
        previous = np.median(levels[i - 22:i - 2], axis=0)
        if np.min(levels[i] - previous) < 12:
            continue
        following = np.median(levels[i + 3:i + 12], axis=0)
        if np.min(levels[i] - following) >= 6:
            hits.append(i)
    # Several adjacent transients often belong to the same acoustic impulse.
    return [int(group[0]) * FRAME / RATE for group in np.split(hits, np.flatnonzero(np.diff(hits) > 8) + 1)
            if len(group)]
