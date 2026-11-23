"""Night-time light tracks and edited changes of field of view."""
import cv2
import numpy as np

from video_frames import samples


def rear_lights(frame):
    blue, green, red = cv2.split(frame.astype(np.int16))
    mask = (red > 105) & (red > green * 1.45 + 15) & (red > blue * 1.45 + 15)
    h, w = mask.shape
    mask[:int(.27 * h)] = mask[int(.81 * h):] = False
    binary = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    _, _, stats, centres = cv2.connectedComponentsWithStats(binary)
    return [(x, y, area) for (x, y), (_, _, bw, bh, area) in zip(centres[1:], stats[1:])
            if 14 <= area <= 1800 and bw < .13 * w and bh < .14 * h]


def lost_light_events(frames):
    lights = [rear_lights(frame) for _, frame in frames]
    tracks = []
    for i, points in enumerate(lights):
        matches = []
        for k, track in enumerate(tracks):
            last, x, y, _ = track[-1]
            if i - last > 2:
                continue
            vx = vy = 0
            if len(track) > 1:
                prev, px, py, _ = track[-2]
                vx, vy = (x - px) / (last - prev), (y - py) / (last - prev)
            for j, (nx, ny, _) in enumerate(points):
                gap = i - last
                error = np.hypot(nx - x - vx * gap, 1.6 * (ny - y - vy * gap))
                if error < 21 * gap and np.hypot(nx - x, ny - y) < 33 * gap:
                    matches.append((error, k, j))
        used_tracks, used_points = set(), set()
        for _, k, j in sorted(matches):
            if k not in used_tracks and j not in used_points:
                tracks[k].append((i, *points[j]))
                used_tracks.add(k)
                used_points.add(j)
        tracks.extend([[(i, *p)] for j, p in enumerate(points) if j not in used_points])

    events = []
    for track in tracks:
        last, x, y, _ = track[-1]
        if len(track) < 7 or last + 9 >= len(lights) or not 90 < x < 550:
            continue
        recent = [p for p in track if p[0] >= last - 14]
        if len(recent) < 7 or abs(x - recent[0][1]) < 30:
            continue
        # The track slows in the road, disappears, then nearby brake lights brighten.
        if np.hypot(x - recent[-3][1], y - recent[-3][2]) > 14:
            continue
        # Compression can split the same nearly stopped lamp into adjacent blobs.
        # Follow its last visible pixels before testing that it disappeared.
        end = last
        for i in range(last + 1, min(last + 7, len(lights))):
            if i - end > 2:
                break
            if any(np.hypot(px-x, py-y) < 13 for px, py, _ in lights[i]):
                end = i
        if end + 9 >= len(lights) or any(np.hypot(px-x, py-y) < 13
                for row in lights[end+1:end+6] for px, py, _ in row):
            continue
        def nearby_peak(row):
            return max((a for px, py, a in row if abs(px-x) <= 75 and abs(py-y) <= 33), default=0)
        prior = np.median([nearby_peak(row) for row in lights[max(0, last-5):last+1]])
        after = [nearby_peak(row) for row in lights[end+2:end+20]]
        if sum(a >= max(180, prior * 2.5) for a in after) >= 3:
            events.append({"time": frames[end+1][0], "status": "light_track",
                           "signal": "Обрыв траектории и торможение рядом"})
    return events


def night_events(path, start, duration):
    events = []
    for offset in range(0, int(np.ceil(duration)), 12):
        span = min(18, duration - offset)
        frames = samples(path, start + offset, span, 6, 640, 360)
        if frames and np.median(frames[len(frames)//2][1]) < 65:
            events.extend(lost_light_events(frames))
        if offset + span >= duration:
            break
    return events


def edited_zoom(path, moment, duration):
    # A cut between a zoomed crop and the original image makes the entire
    # background change scale consistently. Real local motion rarely does.
    start = max(0, moment - .5)
    frames = samples(path, start, min(2.4, duration - start), 5, 640, 360)
    sift = cv2.SIFT_create(nfeatures=900, contrastThreshold=.012)
    features = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None) for _, im in frames]
    for (a, da), (b, db) in zip(features, features[3:]):
        if da is None or db is None or min(len(da), len(db)) < 12:
            continue
        pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
        good = [m for m, n in pairs if m.distance < .77 * n.distance]
        if len(good) < 10:
            continue
        src = np.float32([a[m.queryIdx].pt for m in good])
        dst = np.float32([b[m.trainIdx].pt for m in good])
        matrix, inliers = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=2)
        if matrix is None or inliers.sum() < 12 or inliers.mean() < .20:
            continue
        spread = np.ptp(src[inliers.ravel() != 0], axis=0)
        scale = np.hypot(matrix[0, 0], matrix[1, 0])
        if (scale < .77 or scale > 1.30) and abs(matrix[1, 0]) < .05 and spread[0] > 235 and spread[1] > 96:
            return True
    return False
