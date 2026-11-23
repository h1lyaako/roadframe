"""Geometric proposals for fixed roadside cameras. Coordinates refer to the full frame."""

import math

import cv2
import numpy as np

from video_frames import crop_region, samples


def vehicle_meeting(one, two, vel1, vel2, step):
    """Extrapolate the closest approach from two incoming vehicle-sized tracks."""
    for box, velocity in ((one, vel1), (two, vel2)):
        if box[2] < 0.7 * box[3] or np.linalg.norm(velocity) < max(1.5, box[2] * .09):
            return None
    relative = vel1 - vel2
    distance = one[:2] - two[:2]
    speed2 = float(relative @ relative)
    if speed2 < 1:
        return None
    steps = -float(distance @ relative) / speed2
    offset = (steps - 1) * step  # The last separate boxes occurred one sample ago.
    if not -step <= offset <= 1.0:
        return None
    separation = distance + steps * relative
    # Corners of the bounding rectangles often overlap in neighbouring lanes.
    # Require the projected centre to enter the other vehicle's footprint.
    if all(np.linalg.norm(2 * separation / box[2:]) > 1 for box in (one, two)):
        return None
    return offset


def encounters(frames, vehicle_tracks=False):
    if len(frames) < 8:
        return []
    gray = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for _, img in frames])
    background = np.median(gray[::3], axis=0).astype(np.uint8)
    masks = (
        cv2.absdiff(
            gray.reshape(-1, gray.shape[2]),
            np.broadcast_to(background, gray.shape).reshape(-1, gray.shape[2]),
        ).reshape(gray.shape)
        > 18
    )
    if np.median(masks.mean(axis=(1, 2))) > 0.10:
        return []
    h, w = gray.shape[1:]
    history = []
    candidates = []
    for j, mask in enumerate(masks):
        mask[: int(0.06 * h)] = False
        binary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        boxes = []
        for x, y, bw, bh, area in stats[1:]:
            if 6 <= bw <= 0.33 * w and 4 <= bh <= 0.36 * h and area >= 24 and area / (bw * bh) >= 0.20:
                boxes.append(np.array([x + bw / 2, y + bh / 2, bw, bh]))
        history.append(boxes)
        if j < 3:
            continue
        for current in boxes:
            cx, cy, bw, bh = current
            inside = []
            for previous in history[j - 1]:
                displacement = current[:2] - previous[:2]
                if np.any(np.abs(displacement) > current[2:] * 0.62 + 2):
                    continue
                ratio = np.prod(previous[2:]) / np.prod(current[2:])
                if not 0.14 < ratio < 0.9:
                    continue
                # Estimate an approach from an older nearby box of similar dimensions.
                origins = [
                    b
                    for b in history[j - 3]
                    if np.all(np.abs(b[:2] - previous[:2]) < previous[2:] * 1.65 + 3)
                    and np.max(np.abs(np.log(b[2:] / previous[2:]))) < 0.60
                ]
                if origins:
                    origin = min(
                        origins,
                        key=lambda b: np.linalg.norm(b[:2] - previous[:2])
                        + 6 * np.sum(np.abs(np.log(b[2:] / previous[2:]))),
                    )
                    inside.append((previous, (previous[:2] - origin[:2]) / 2))
            for a, (one, vel1) in enumerate(inside):
                for two, vel2 in inside[a + 1 :]:
                    sep = one[:2] - two[:2]
                    distance = np.linalg.norm(sep)
                    if distance < 4 or np.any(np.abs(sep) > 0.8 * (one[2:] + two[2:]) + 3):
                        continue
                    closing = -np.dot(sep, vel1 - vel2) / distance
                    score = (
                        closing
                        / max(4, min(one[2], two[2]))
                        * min(1.0, min(np.prod(one[2:]), np.prod(two[2:])) / 90)
                    )
                    if score > 0.055:
                        timestamp = frames[j][0]
                        meeting = timestamp
                        if vehicle_tracks:
                            offset = vehicle_meeting(one, two, vel1, vel2, timestamp - frames[j - 1][0])
                            meeting = None if offset is None else timestamp + offset
                        candidates.append(
                            (
                                round(float(score), 3),
                                round(timestamp, 3),
                                (cx / w, cy / h),
                                max(0.20, min(0.75, 4 * max(bw / w, bh / h))),
                                meeting,
                            )
                        )
    kept = []
    for item in sorted(candidates, key=lambda candidate: candidate[:4], reverse=True):
        if not any(abs(item[1] - p[1]) < 2 and math.dist(item[2], p[2]) < 0.22 for p in kept):
            kept.append(item)
    # Rank observed encounters before interpreting their trajectories. Rejecting
    # a strong ordinary pass should not promote weaker background coincidences.
    if vehicle_tracks:
        kept = kept[:4]
    return [(score, round(meeting, 3), centre, size)
            for score, _, centre, size, meeting in kept if meeting is not None]


def persistent_change(path, duration, anchor, region):
    """Test whether a new object or debris remains after a fixed-camera encounter."""
    start = max(0, anchor - 6)
    frames = samples(path, start, min(duration - start, anchor + 7 - start), 2, 448, 252)
    x, y, size = region
    patch = crop_region(frames, (x, y, min(1, size * 1.45))) if frames else []
    before = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for t, im in patch if t < anchor - 0.65]
    after = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for t, im in patch if t > anchor + 1.5]
    if min(len(before), len(after)) < 3:
        return False
    low1, high1 = np.quantile(before, [0.2, 0.8], axis=0)
    low2, high2 = np.quantile(after, [0.15, 0.85], axis=0)
    changed = (np.maximum(low1 - high2, low2 - high1) > 20).astype(np.uint8)
    joined = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    return changed.mean() >= 0.007 and np.max(stats[1:, 4], initial=0) >= 0.004 * changed.size


def roadside_events(path, duration, progress):
    events = []
    checked = []
    for start in range(0, math.ceil(duration), 14):
        frames = samples(path, start, min(16, duration - start), 5, 640, 360)
        for score, moment, (x, y), size in encounters(frames, vehicle_tracks=True):
            if score < .12 or not (.06 < x < .94 and .1 < y < .9):
                continue
            # Context on both sides distinguishes a new obstruction from passing traffic.
            if moment < 3 or duration - moment < 5:
                continue
            if any(abs(moment - t) < 1 and math.dist((x, y), xy) < .08 for t, xy in checked):
                continue
            checked.append((moment, (x, y)))
            region = (x, y, min(.4, max(.26, size * .7)))
            if persistent_change(path, duration, moment, region):
                events.append({"time": round(moment, 2), "signal": "Возможное столкновение"})
        progress(min(1, (start + 16) / duration))
        if start + 16 >= duration:
            break
    return events
