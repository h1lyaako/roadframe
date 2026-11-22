"""Geometric proposals for fixed roadside cameras. Coordinates refer to the full frame."""

import math

import cv2
import numpy as np


def encounters(frames):
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
                        candidates.append(
                            (
                                round(float(score), 3),
                                frames[j][0],
                                (cx / w, cy / h),
                                max(0.20, min(0.75, 4 * max(bw / w, bh / h))),
                            )
                        )
    kept = []
    for item in sorted(candidates, reverse=True):
        if not any(abs(item[1] - p[1]) < 2 and math.dist(item[2], p[2]) < 0.22 for p in kept):
            kept.append(item)
    return kept
