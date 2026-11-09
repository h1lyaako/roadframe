"""Temporal collision candidates from dashcam video."""
import json
import subprocess

import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 256, 144, 5
MAX_DURATION = 3600
cv2.setNumThreads(2)


def probe(path):
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name:format=duration", "-of", "json", str(path)
    ], timeout=20, stderr=subprocess.DEVNULL)
    data = json.loads(output)
    duration = float(data["format"]["duration"])
    if not data.get("streams") or not 1 <= duration <= MAX_DURATION:
        raise ValueError("Нужна видеозапись длительностью от секунды до часа.")
    return duration, data["streams"][0]["codec_name"]


def frame_signals(previous, current, last_motion):
    points = cv2.goodFeaturesToTrack(previous, 110, .02, 5)
    motion = None
    matrix = None
    if points is not None:
        tracked, valid, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
        if tracked is not None and valid is not None:
            mask = valid.ravel() == 1
            if mask.sum() >= 8:
                matrix, inliers = cv2.estimateAffinePartial2D(
                    points[mask], tracked[mask], method=cv2.RANSAC, ransacReprojThreshold=2
                )
                if matrix is not None:
                    scale = np.linalg.norm(matrix[0, :2])
                    angle = abs(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
                    if (inliers.sum() < 8 or inliers.mean() < .45 or not .88 <= scale <= 1.12
                            or angle > 12 or np.linalg.norm(matrix[:, 2]) > 56):
                        matrix = None
    if matrix is not None:
        motion = matrix[:, 2]
        previous = cv2.warpAffine(previous, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REPLICATE)
    # Ignore the outer frame, dashboard and sky. Use the strongest local region.
    diff = cv2.absdiff(previous, current)[24:120, 24:232]
    cells = (diff > 27).reshape(4, 24, 8, 26).mean(axis=(1, 3))
    change = float(np.mean(np.sort(cells.ravel())[-4:]))
    jerk = float(np.linalg.norm(motion - last_motion)) if motion is not None and last_motion is not None else 0.0
    return (change, jerk), motion


def decoded_frames(path, fps):
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-i", str(path),
        "-an", "-sn", "-vf", f"fps={fps},scale={WIDTH}:{HEIGHT},format=gray",
        "-threads", "1", "-f", "rawvideo", "pipe:1"
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as decoder:
        try:
            while raw := decoder.stdout.read(WIDTH * HEIGHT):
                if len(raw) != WIDTH * HEIGHT:
                    break
                yield np.frombuffer(raw, np.uint8).reshape(HEIGHT, WIDTH)
            if decoder.wait(timeout=10) != 0:
                raise ValueError("Не удалось прочитать кадры записи.")
        finally:
            if decoder.poll() is None:
                decoder.kill()


def analyze_baseline(path, duration, update=lambda fraction: None):
    signals = []
    previous = None
    motion = None
    for index, frame in enumerate(decoded_frames(path, FPS)):
        if previous is not None:
            signal, motion = frame_signals(previous, frame, motion)
            signals.append(signal)
            if index % (FPS * 2) == 0:
                update(min(1, index / FPS / duration))
        previous = frame
    if not signals:
        raise ValueError("Не удалось прочитать кадры записи.")
    values = np.asarray(signals)
    median = np.median(values, axis=0)
    spread = 1.4826 * np.median(abs(values - median), axis=0)
    scores = np.maximum(0, (values - median) / np.maximum(spread, [.055, 1.5]))
    strength = scores @ np.array([.65, .35])
    events = []
    for index in np.argsort(-strength):
        seconds = (int(index) + 1) / FPS
        if strength[index] < 2.1 or len(events) == 12:
            break
        if any(abs(item["time"] - seconds) < 4 for item in events):
            continue
        label = "Рывок камеры" if scores[index, 1] * .35 > scores[index, 0] * .65 else "Резкое движение в кадре"
        events.append({"time": round(seconds, 2), "signal": label})
    return events


IMPACT_FPS = 10
# Look at the roadway, with the outside image border and the dashboard removed.
ROAD = (slice(30, 126), slice(32, 224))
GRID_Y, GRID_X = np.mgrid[:HEIGHT:4, :WIDTH:4].astype(np.float32)
GRID_POINTS = np.column_stack((GRID_X.ravel(), GRID_Y.ravel()))
ROAD_X, ROAD_Y = np.meshgrid(np.arange(32, 224, dtype=np.float32), np.arange(30, 126, dtype=np.float32))


def motion_fields(path, duration, update=lambda fraction: None):
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    previous = None
    previous_camera = None
    local_fields = []
    camera_jerks = []
    background_motion = []
    texture = []
    for index, frame in enumerate(decoded_frames(path, IMPACT_FPS)):
        # A hood or a nearby obstacle can fill the roadway and make optical
        # flow unreliable. Measure detail in the road and sky independently.
        texture.append((
            float(np.abs(cv2.Laplacian(frame[70:122, 64:192], cv2.CV_32F)).mean()),
            float(frame[26:56, 52:204].mean()),
        ))
        # Equalization removes most global exposure jumps before measuring movement.
        frame = cv2.equalizeHist(frame)
        if previous is not None:
            displacement = flow.calc(previous, frame, None)
            samples = displacement[::4, ::4].reshape(-1, 2)
            background_motion.append(float(np.median(np.linalg.norm(samples, axis=1))))
            matrix, inliers = cv2.estimateAffinePartial2D(
                GRID_POINTS, GRID_POINTS + samples, method=cv2.RANSAC,
                ransacReprojThreshold=1.35, maxIters=100
            )
            valid = matrix is not None and inliers.mean() >= .45
            if valid:
                # The transformed image centre measures camera motion without
                # counting rotation around the centre as a lateral translation.
                camera = matrix @ np.array([WIDTH / 2, HEIGHT / 2, 1.0])
                camera -= [WIDTH / 2, HEIGHT / 2]
                x, y = ROAD_X, ROAD_Y
                global_flow = np.stack((
                    (matrix[0, 0] - 1) * x + matrix[0, 1] * y + matrix[0, 2],
                    matrix[1, 0] * x + (matrix[1, 1] - 1) * y + matrix[1, 2],
                ), axis=-1)
                residual = displacement[ROAD] - global_flow
            else:
                camera = None
                residual = displacement[ROAD] - np.median(samples, axis=0)
            # Use coherent movement of regions. Isolated optical-flow errors
            # and diffuse camera movement have a weaker effect on these cells.
            cells = residual.reshape(6, 16, 12, 16, 2).mean(axis=(1, 3))
            local_fields.append(cells.astype(np.float32))
            jerk = np.linalg.norm(camera - previous_camera) if valid and previous_camera is not None else 0
            camera_jerks.append(jerk)
            previous_camera = camera
            if index % (IMPACT_FPS * 2) == 0:
                update(min(1, index / IMPACT_FPS / duration))
        previous = frame
    if not local_fields:
        raise ValueError("Не удалось прочитать кадры записи.")
    return np.asarray(local_fields), np.asarray(camera_jerks), np.asarray(background_motion), np.asarray(texture)


def occlusion_intervals(texture, local_motion):
    detail, upper_luminance = texture.T
    intervals = []
    before, after = 8, 14
    for index in range(before, len(detail) - after):
        if intervals and index <= intervals[-1][1]:
            continue
        # Check the first low-detail frame against the stable scene preceding
        # it. Reject short blur, camera fades and global light changes.
        if detail[index] > detail[index - 2] * .65:
            continue
        prior = float(np.median(detail[index - before:index - 1]))
        if prior < 8 or detail[index] > prior * .50:
            continue
        if np.quantile(detail[index:index + after], .75) > prior * .45:
            continue
        upper_before = np.median(upper_luminance[index - before:index])
        upper_after = np.median(upper_luminance[index:index + after])
        if upper_after < 20 or abs(upper_before - upper_after) > 30:
            continue
        movement = local_motion[max(0, index - 5):index + 5]
        if movement.max(initial=0) < 4:
            continue
        # Suppress motion guesses while the road is still obstructed.
        end = index + after
        while end + 5 < len(detail) and np.median(detail[end:end + 5]) < prior * .65:
            end += 1
        intervals.append((index, end))
    return intervals


def collision_candidates(fields, camera_jerks, background_motion, texture):
    speed = np.linalg.norm(fields, axis=-1)
    impulse = np.linalg.norm(np.diff(fields, axis=0, prepend=fields[:1]), axis=-1)
    # A region must actually move and then change speed or direction.
    local = np.sqrt(speed * impulse).reshape(len(fields), -1)
    local = np.sort(local, axis=1)[:, -4:].mean(axis=1)
    # A candidate needs support in nearby frames, not a single corrupted frame.
    supported = np.convolve(np.pad(local, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
    baseline = float(np.median(supported))
    score = supported + .25 * np.minimum(camera_jerks, 12)
    stationary = float(np.quantile(background_motion, .75)) < .4
    # On a parked camera, background motion contributes very little to the
    # noise floor. A smaller sustained disturbance can then be meaningful.
    threshold = max(3.8, baseline * 2.65) if stationary else max(8.0, baseline * 3.6)
    # Join neighboring bursts so the aftermath of one crash doesn't create
    # several marks. Report the start of the first confirmed burst.
    active = np.flatnonzero(score >= threshold)
    groups = np.split(active, np.where(np.diff(active) > round(2.5 * IMPACT_FPS))[0] + 1)
    events = []
    for group in groups:
        if len(group) < 2 or not np.any(np.diff(group) == 1):
            continue
        onset = int(group[np.flatnonzero(np.diff(group) == 1)[0]])
        quiet_limit = max(2.5, baseline, threshold * .28)
        quiet = 0
        for index in range(onset - 1, max(-1, onset - 15), -1):
            if supported[index] >= quiet_limit:
                onset = index
                quiet = 0
            else:
                quiet += 1
                if quiet == 3:
                    break
        events.append((float(score[group].max()), {
            "time": round((onset + 1) / IMPACT_FPS, 2),
            "signal": "Возможное столкновение",
        }))
    occlusions = occlusion_intervals(texture, local)
    # One mark for the same scene change, favoring the observed occlusion
    # over weaker optical-flow bursts in the immediate neighborhood.
    events = [
        (score, event) for score, event in events
        if all(not index / IMPACT_FPS - 3.5 <= event["time"] <= end / IMPACT_FPS
               for index, end in occlusions)
    ]
    events.extend((float("inf"), {
        "time": round(index / IMPACT_FPS, 2),
        "signal": "Возможное столкновение",
    }) for index, _ in occlusions)
    return [event for _, event in sorted(events, key=lambda item: -item[0])[:12]]


def analyze(path, duration, update=lambda fraction: None):
    return collision_candidates(*motion_fields(path, duration, update))
