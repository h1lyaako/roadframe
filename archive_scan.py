"""Full-length packet survey, keyframe overview and short high-detail windows."""
import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from analysis import analyze


SPAN = 20
MAX_WINDOWS = 4


def novelty(descriptors, times):
    # Compare short sequences rather than isolated pictures or individual packet spikes.
    # Repeated scenery, exposure cycles and periodic encoder changes get lower priority.
    features = np.ascontiguousarray(descriptors, dtype=np.float32)
    matches = cv2.BFMatcher().knnMatch(features, features, k=min(16, len(features)))
    return np.array([next((m.distance for m in row if abs(times[m.trainIdx] - times[i]) > 15),
                          row[-1].distance) for i, row in enumerate(matches)])


def packet_overview(path, duration):
    data = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "V:0", "-show_entries",
        "packet=pts_time,size,flags", "-of", "csv=p=0", str(path),
    ], timeout=18, stderr=subprocess.DEVNULL)
    packets = []
    for row in data.splitlines():
        parts = row.split(b",")
        if len(parts) >= 3 and parts[0] not in (b"", b"N/A") and parts[1].isdigit():
            packets.append((float(parts[0]), int(parts[1]), b"K" in parts[2]))
    if not packets:
        raise ValueError("Не удалось построить временную шкалу видеодорожки.")
    pts, sizes, keys = np.asarray(packets, dtype=np.float64).T
    origin = pts.min()
    pts -= origin
    bins = np.clip(pts.astype(int), 0, math.ceil(duration) - 1)
    selected = keys == 0
    if selected.sum() < len(keys) / 4:
        selected = np.ones(len(keys), dtype=bool)
    counts = np.bincount(bins[selected], minlength=math.ceil(duration))
    totals = np.bincount(bins[selected], weights=sizes[selected], minlength=len(counts))
    levels = np.log1p(totals / np.maximum(counts, 1))
    levels = np.interp(np.arange(len(levels)), np.flatnonzero(counts), levels[counts > 0])
    local = np.lib.stride_tricks.sliding_window_view(np.pad(levels, 7, mode="edge"), 15)
    scores = np.abs(levels - np.median(local, axis=1)) + .35 * np.abs(np.diff(levels, prepend=levels[0]))
    scores = (scores + .35) * (.18 + np.minimum(3, novelty(local, np.arange(len(levels)))))
    return scores, np.sort(pts[keys != 0]), origin


def visual_overview(path, key_times):
    if len(key_times) < 2:
        return None, key_times
    stride = math.ceil(len(key_times) / 2000)
    key_times = key_times[::stride]
    # Drop dependent packets before decoding. Asking the decoder to skip them
    # still makes it parse a full hour of inter frames on some codecs.
    drop = "not(key)" if stride == 1 else f"st(0,ld(0)+key);not(key*not(mod(ld(0)-1,{stride})))"
    try:
        with tempfile.TemporaryDirectory(prefix="roadframe-keys-") as folder:
            keys = Path(folder) / "keys.mkv"
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:V:0", "-c", "copy",
                "-bsf:v", f"noise=drop='{drop}'", str(keys),
            ], timeout=10, check=True, capture_output=True)
            raw = subprocess.check_output([
                "ffmpeg", "-v", "error", "-threads", "6", "-skip_loop_filter", "all", "-i", str(keys),
                "-map", "0:V:0", "-vf", "scale=48:27:flags=fast_bilinear,format=gray",
                "-fps_mode", "passthrough", "-threads", "1", "-f", "rawvideo", "pipe:1",
            ], timeout=14, stderr=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError):
        return None, key_times
    if len(raw) != len(key_times) * 48 * 27:
        return None, key_times
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 27, 48)
    descriptors = np.float32([cv2.resize(im[4:25], (8, 5), interpolation=cv2.INTER_AREA).ravel() for im in frames])
    changes = np.quantile(np.abs(np.diff(descriptors, axis=0, prepend=descriptors[:1])), .82, axis=1)
    transitions = np.column_stack((np.vstack((descriptors[:1], descriptors[:-1])), descriptors))
    distances = novelty(transitions, key_times)
    return (changes + 12) * (.18 + np.minimum(distances / 60, 4)), key_times


def audio_overview(path, duration, origin):
    try:
        raw = subprocess.check_output([
            "ffmpeg", "-v", "error", "-copyts", "-i", str(path), "-map", "0:a:0", "-vn",
            "-af", f"asetpts=PTS-({origin})/TB,aresample=4000:async=1:first_pts=0,highpass=f=700",
            "-ac", "1", "-t", str(duration), "-f", "s16le", "pipe:1",
        ], timeout=12, stderr=subprocess.DEVNULL)
    except subprocess.SubprocessError:
        return []
    signal = np.frombuffer(raw, "<i2")
    count = len(signal) // 500
    if count < 18:
        return []
    # Eight energy readings per second; only a sudden rise over the local floor
    # nominates a window. Detailed analysis will check motion and the original audio.
    power = np.square(signal[:count*500].reshape(-1, 500), dtype=np.float32).mean(axis=1)
    db = 10 * np.log10(1 + power)
    history = np.lib.stride_tricks.sliding_window_view(np.pad(db, (16, 0), mode="edge"), 17)
    rise = db - np.median(history[:, :14], axis=1)
    peaks = np.flatnonzero((rise > 14) & (db > 45))
    return peaks[np.argsort(-rise[peaks], kind="stable")] / 8


def window_starts(scores, key_times, visual, duration, sounds):
    # Each modality gets a turn so exposure transitions cannot consume every slot.
    rankings = [np.argsort(-scores, kind="stable")]
    if visual is not None:
        rankings.insert(0, key_times[np.argsort(-visual, kind="stable")])
    if len(sounds):
        rankings.insert(1 if visual is not None else 0, sounds)
    starts = []
    for rank in range(max(len(r) for r in rankings)):
        for times in rankings:
            if rank >= len(times):
                continue
            start = min(max(0., float(times[rank]) - 5), max(0., duration - SPAN))
            if all(abs(start - previous) >= SPAN for previous in starts):
                starts.append(start)
            if len(starts) >= MAX_WINDOWS:
                return starts
    return starts


def scan_archive(path, duration, progress):
    scores, key_times, origin = packet_overview(path, duration)
    progress("Сравниваем опорные кадры", 23)
    with ThreadPoolExecutor(max_workers=2) as pool:
        audio = pool.submit(audio_overview, path, duration, origin)
        visual, key_times = visual_overview(path, key_times)
        starts = window_starts(scores, key_times, visual, duration, audio.result())
    progress("Проверяем выбранные фрагменты", 48)
    events = []
    with tempfile.TemporaryDirectory(prefix="roadframe-windows-") as folder:
        def inspect(item):
            index, start = item
            clip = Path(folder) / f"window-{index}.mkv"
            span = min(SPAN, duration - start)
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-threads", "3", "-i", str(path),
                "-t", str(span), "-map", "0:V:0", "-map", "0:a:0?", "-vf",
                "fps=12,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264", "-threads", "2", "-preset", "ultrafast", "-crf", "18",
                "-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000", str(clip),
            ], timeout=18, check=True, capture_output=True)
            found = analyze(clip, span)
            for e in found:
                e["time"] = round(e["time"] + start, 2)
            return found

        with ThreadPoolExecutor(max_workers=2) as pool:
            for i, found in enumerate(pool.map(inspect, enumerate(starts))):
                events.extend(found)
                progress("Проверяем выбранные фрагменты", 48 + round(48 * (i + 1) / len(starts)))
    return sorted(events, key=lambda e: e["time"]), {
        "kind": "indexed", "seconds": round(duration, 2), "keyframes": len(key_times) if visual is not None else 0,
        "windows": [[round(t, 2), round(min(t + SPAN, duration), 2)] for t in sorted(starts)],
    }
