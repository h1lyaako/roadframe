"""Temporal collision candidates from dashcam video."""
import json
import math
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np
from encounters import roadside_events

WIDTH, HEIGHT, FPS = 256, 144, 5
MAX_DURATION = 3600
cv2.setNumThreads(2)


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    codec: str
    browser_ready: bool


def video_info(path):
    try:
        output = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,codec_name,pix_fmt,duration,start_time:stream_disposition=attached_pic:"
            "stream_tags=DURATION:format=duration",
            "-of", "json", str(path)
        ], timeout=20, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise ValueError("Не удалось прочитать видеофайл.") from exc
    data = json.loads(output)
    streams = data.get("streams", [])
    videos = [s for s in streams if s["codec_type"] == "video" and not s["disposition"]["attached_pic"]]
    if not videos:
        raise ValueError("В файле нет видеодорожки.")
    video = videos[0]
    container = float(data["format"].get("duration", 0))
    if video.get("duration"):
        duration = float(video["duration"])
    elif tag := video.get("tags", {}).get("DURATION"):
        h, m, s = map(float, tag.split(":"))
        duration = h * 3600 + m * 60 + s - float(video.get("start_time", 0))
    elif len(streams) > 1 or not container:
        # Some muxers only store a container duration. A second track can outlast the first.
        packets = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "V:0", "-show_entries",
            "packet=pts_time,duration_time", "-of", "compact=p=0:nk=0", str(path)
        ], timeout=90, stderr=subprocess.DEVNULL)
        first, last = math.inf, -math.inf
        for line in packets.decode("ascii").splitlines():
            values = dict(pair.split("=", 1) for pair in line.split("|") if "=" in pair)
            if values.get("pts_time", "N/A") == "N/A":
                continue
            pts = float(values["pts_time"])
            length = values.get("duration_time", "N/A")
            first, last = min(first, pts), max(last, pts + (0 if length == "N/A" else float(length)))
        duration = last - first
    else:
        duration = container
    if not 1 <= duration <= MAX_DURATION:
        raise ValueError("Нужна видеозапись длительностью от секунды до часа.")
    browser_ready = (len(videos) == 1 and video["codec_name"] == "h264"
                     and video.get("pix_fmt") in {"yuv420p", "yuvj420p"}
                     and abs(container - duration) < 0.1
                     and all(s["codec_name"] in {"aac", "mp3"} for s in streams if s["codec_type"] == "audio"))
    return VideoInfo(duration, video["codec_name"], browser_ready)


def probe(path):
    info = video_info(path)
    return info.duration, info.codec


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
        "-map", "0:V:0", "-an", "-sn", "-vf", f"fps={fps},scale={WIDTH}:{HEIGHT},format=gray",
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
# Directions from the image centre to the centres of the 6 x 12 road cells.
CELL_X, CELL_Y = np.meshgrid(np.arange(40, 224, 16) - WIDTH / 2, np.arange(38, 126, 16) - HEIGHT / 2)
CELL_RAYS = np.stack((CELL_X, CELL_Y), axis=-1)
CELL_RAYS = CELL_RAYS / np.linalg.norm(CELL_RAYS, axis=-1, keepdims=True)


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
        # A car crossing several grid cells creates a large impulse at each cell edge
        # even at constant velocity. Test whether the strongest motion stays parallel.
        peak = int(group[np.argmax(score[group])])
        window = fields[max(0, peak - 4):peak + 5]
        lengths = np.linalg.norm(window, axis=-1)
        strong = lengths >= max(2.0, np.quantile(lengths, .90))
        moving = window[strong]
        jerk = np.max(camera_jerks[group])
        if len(moving) and jerk < 3:
            alignment = np.linalg.norm(moving.sum(axis=0)) / np.linalg.norm(moving, axis=1).sum()
            if alignment > .985:
                continue
            # Forward travel moves scenery outwards on both sides. A nearby
            # wall or tree can also dominate the outer columns on just one side.
            if jerk < 1:
                unit = moving / lengths[strong][:, None]
                rays = np.broadcast_to(CELL_RAYS, window.shape)[strong]
                outward = (unit * rays).sum(axis=1) > np.cos(np.pi / 4)
                both_sides = all((outward & side).mean() >= .1 for side in (rays[:, 0] < 0, rays[:, 0] > 0))
                edge = np.broadcast_to(np.abs(CELL_X) > WIDTH / 5, strong.shape)[strong]
                if outward.mean() >= .8 and (both_sides or (edge & outward).mean() >= .75):
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
    fields = motion_fields(path, duration, lambda fraction: update(.8 * fraction))
    events = collision_candidates(*fields)
    # A genuinely fixed frame permits tracking vehicles against a stable background.
    # A stopped dashcam with engine vibration usually exceeds this noise floor.
    if np.quantile(fields[2], .75) < .05:
        for candidate in roadside_events(path, duration, lambda fraction: update(.8 + .2 * fraction)):
            if all(abs(candidate["time"] - e["time"]) >= 2 for e in events):
                events.append(candidate)
    update(1)
    return sorted(events, key=lambda event: event["time"])
