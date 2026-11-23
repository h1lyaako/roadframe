import argparse
import math
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

from analysis import analyze, video_info
from archive_scan import scan_archive
from vision import review
from codex_review import review_codex

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "static" / "demo.mp4"
DEMO_PARTS = tuple(ROOT / "static" / "demo" / f"clip-{index:02}.mp4" for index in range(1, 4))
RU_DEMO = (ROOT / "test-videos-ru" / "01-trekhgorny.webm", ROOT / "test-videos-ru" / "02-ozersk.webm")
COLLISION_DEMO = (ROOT / "test-videos-ru" / "04-saint-petersburg.mp4",
                  ROOT / "test-videos-ru" / "07-trekhgorny-control.mp4")
WORK = tempfile.TemporaryDirectory(prefix="roadframe-")
app = Flask(__name__)
MAX_VIDEO_BYTES = 1024 ** 3
app.config["MAX_CONTENT_LENGTH"] = MAX_VIDEO_BYTES + 1024 ** 2
jobs = {}
lock = threading.Lock()


def transcode(source, target, selection=None, duration=None):
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if selection:
        args += ["-ss", str(selection[0])]
    args += ["-i", str(source)]
    if selection:
        args += ["-t", str(selection[1] - selection[0])]
        # Chroma subsampling needs even dimensions. Preserve the original picture.
        args += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    else:
        if duration is not None:
            args += ["-t", str(duration)]
        args += ["-vf", "scale='min(960,iw)':-2,pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    args += [
        "-map", "0:V:0", "-map", "0:a:0?", "-c:v", "libx264", "-threads", "4",
        "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-movflags", "+faststart", str(target),
    ]
    subprocess.run(args, check=True, timeout=900, capture_output=True)


def update(job, **values):
    with lock:
        job["view"].update(values)


def process(job):
    try:
        source = job["source"]
        info = video_info(source)
        duration = info.duration
        large = duration >= 120 or source.stat().st_size >= 100 * 1024 ** 2
        if large:
            update(job, phase="Читаем индекс всей видеодорожки")
            events, scan = scan_archive(source, duration, lambda phase, value: update(job, phase=phase, progress=value))
            update(job, scan=scan, effectiveMode="motion")
        elif job["view"]["mode"] == "codex":
            result = review_codex(source, duration, job["folder"], lambda phase, value:
                                  update(job, phase=phase, progress=round(value)))
            events = result.pop("events")
            update(job, **result)
        elif job["view"]["mode"] == "ru":
            events, elapsed = review(source, duration, lambda phase, value:
                                     update(job, phase=phase, progress=round(value)))
            update(job, visionSeconds=elapsed)
        else:
            events = analyze(source, duration, lambda fraction: update(
                job, progress=round(fraction * 95)
            ))
        if info.native_mime:
            job["preview"] = source
            job["mime"] = info.native_mime
        elif large:
            update(job, previewType="hls", phase="Готовим видео для просмотра")
            # Prime the first requested position. The whole original stays seekable.
            preview_part(job, int((events[0]["time"] if events else 0) // PART_SECONDS), duration)
        else:
            update(job, phase="Готовим видео для просмотра")
            job["preview"] = job["folder"] / "preview.mp4"
            transcode(source, job["preview"], duration=duration)
        update(job, status="ready", progress=100, duration=duration, events=events)
    except ValueError as error:
        update(job, status="error", error=str(error))
    except Exception:
        app.logger.exception("Video processing failed")
        update(job, status="error", error="Не удалось обработать запись. Попробуйте другой видеофайл.")


def get_job(key):
    with lock:
        job = jobs.get(key)
    if job is None:
        abort(404)
    return job


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Файл превышает лимит 1 ГБ."), 413


@app.post("/api/jobs")
def create_job():
    demo = request.args.get("demo")
    is_demo = demo in {"1", "collision"}
    mode = request.args.get("mode", "motion")
    if mode not in {"ru", "motion", "codex"}:
        return jsonify(error="Неизвестный режим анализа."), 400
    if mode == "codex" and request.headers.get("X-Roadframe-Codex") != "1":
        return jsonify(error="Выберите Codex в приложении, чтобы отправить кадры."), 400
    part = request.args.get("part")
    demo_file = DEMO
    demo_parts = RU_DEMO if mode in {"ru","codex"} else DEMO_PARTS
    if demo == "collision":
        demo_parts = COLLISION_DEMO
    if is_demo and (demo == "collision" or mode in {"ru", "codex"}) and part is None:
        abort(400)
    if is_demo and part is not None:
        if part not in {str(i+1) for i in range(len(demo_parts))}:
            abort(404)
        demo_file = demo_parts[int(part) - 1]
    if not is_demo:
        free = shutil.disk_usage(WORK.name).free
        if free < 1024 ** 3 + (request.content_length or 0) * 2:
            return jsonify(error="Недостаточно свободного места для обработки записи."), 507
    raw = not is_demo and request.mimetype == "application/octet-stream"
    if raw and request.content_length and request.content_length > MAX_VIDEO_BYTES:
        return too_large(None)
    upload = None if is_demo or raw else request.files.get("video")
    name = (request.args.get("name", "") if raw else upload.filename if upload else "") or ""
    if is_demo and not demo_file.is_file():
        return jsonify(error="Демонстрационная запись пока недоступна."), 404
    suffix = Path(name).suffix.lower()
    if not is_demo and suffix not in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
        return jsonify(error="Выберите MP4, MOV, WebM, MKV или AVI."), 400
    key = uuid.uuid4().hex
    folder = Path(WORK.name) / key
    source = demo_file if is_demo else folder / ("source" + suffix)
    demo_name = f"Козе-{int(part):02}.mp4" if is_demo and part is not None else "Козе · зимняя поездка.mp4"
    if is_demo and (demo == "collision" or mode in {"ru", "codex"}):
        demo_name = demo_file.name
    view = {
        "id": key, "status": "working", "phase": "Анализируем движение",
        "progress": 0, "name": demo_name if is_demo else Path(name).name,
        "demo": is_demo, "mode": mode,
    }
    job = {"view": view, "source": source, "folder": folder, "export_lock": threading.Lock()}
    with lock:
        if any(entry["view"]["status"] == "working" for entry in jobs.values()):
            return jsonify(error="Дождитесь завершения текущего анализа."), 409
        jobs[key] = job
    try:
        folder.mkdir()
        if not is_demo:
            with source.open("wb") as output:
                shutil.copyfileobj(request.stream if raw else upload.stream, output, length=1024 ** 2)
            if source.stat().st_size > MAX_VIDEO_BYTES:
                source.unlink()
                with lock:
                    jobs.pop(key, None)
                return too_large(None)
        threading.Thread(target=process, args=(job,), daemon=True).start()
    except HTTPException:
        update(job, status="error", error="Загрузка записи прервалась.")
        raise
    except OSError:
        update(job, status="error", error="Не удалось сохранить запись на диск.")
    return jsonify(id=key), 202


@app.get("/api/jobs/<key>")
def status(key):
    job = get_job(key)
    with lock:
        return jsonify(job["view"])


@app.get("/api/jobs/<key>/video")
def preview(key):
    job = get_job(key)
    if job["view"]["status"] != "ready":
        abort(404)
    if job["view"].get("previewType") == "hls":
        duration = job["view"]["duration"]
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-PLAYLIST-TYPE:VOD",
                 f"#EXT-X-TARGETDURATION:{PART_SECONDS}", "#EXT-X-MEDIA-SEQUENCE:0"]
        for i in range(math.ceil(duration / PART_SECONDS)):
            lines.extend([f"#EXTINF:{min(PART_SECONDS, duration - i * PART_SECONDS):.6f},", f"part/{i}.ts"])
        return Response("\n".join([*lines, "#EXT-X-ENDLIST", ""]), mimetype="application/vnd.apple.mpegurl")
    return send_file(job["preview"], mimetype=job.get("mime", "video/mp4"), conditional=True)


PART_SECONDS = 8


def preview_part(job, i, duration):
    target = job["folder"] / f"part-{i}.ts"
    start = i * PART_SECONDS
    with job["export_lock"]:
        if not target.exists():
            partial = target.with_suffix(".part.ts")
            subprocess.run([
                "ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", str(start), "-threads", "3",
                "-i", str(job["source"]), "-t", str(min(PART_SECONDS, duration - start)),
                "-map", "0:V:0", "-map", "0:a:0?", "-vf",
                "fps=24,scale='min(960,iw)':-2,pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-threads", "3", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-output_ts_offset", str(start), "-f", "mpegts", str(partial),
            ], check=True, timeout=30, capture_output=True)
            partial.replace(target)
    return target


@app.get("/api/jobs/<key>/part/<int:index>.ts")
def part(key, index):
    job = get_job(key)
    view = job["view"]
    if view["status"] != "ready" or view.get("previewType") != "hls" or index * PART_SECONDS >= view["duration"]:
        abort(404)
    return send_file(preview_part(job, index, view["duration"]), mimetype="video/mp2t", conditional=True)


@app.get("/api/jobs/<key>/still")
def still(key):
    job = get_job(key)
    view = job["view"]
    at = request.args.get("at", type=float)
    if view["status"] != "ready" or at is None or not 0 <= at < view["duration"]:
        abort(400)
    data = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", str(at), "-threads", "2", "-i", str(job["source"]),
        "-map", "0:V:0", "-frames:v", "1", "-vf", "scale=160:-2", "-c:v", "mjpeg",
        "-threads", "1", "-f", "image2pipe", "pipe:1",
    ], timeout=15, stderr=subprocess.DEVNULL)
    return Response(data, mimetype="image/jpeg")


@app.get("/api/jobs/<key>/clip")
def export(key):
    job = get_job(key)
    if job["view"]["status"] != "ready":
        abort(409)
    try:
        start = round(float(request.args["start"]), 3)
        end = round(float(request.args["end"]), 3)
        if not (math.isfinite(start) and math.isfinite(end)
                and 0 <= start < end <= job["view"]["duration"] + .001
                and end - start <= 120):
            raise ValueError
    except (KeyError, ValueError, OverflowError):
        return jsonify(error="Укажите отрывок длиной не более двух минут."), 400
    filename = f"clip-{start:.3f}-{end:.3f}.mp4"
    target = job["folder"] / filename
    with job["export_lock"]:
        if not target.exists():
            partial = target.with_suffix(".part.mp4")
            try:
                transcode(job["source"], partial, (start, end))
                partial.replace(target)
            except (OSError, subprocess.SubprocessError):
                partial.unlink(missing_ok=True)
                return jsonify(error="Не удалось сохранить отрывок."), 500
    return send_file(target, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Локальный сервер «Дорожный кадр»")
    parser.add_argument("--port", type=int, default=18065)
    options = parser.parse_args()
    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        raise SystemExit("Добавьте FFmpeg и FFprobe в PATH.")
    app.run(host="127.0.0.1", port=options.port, threaded=True, debug=False)
