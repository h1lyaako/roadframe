"""Visual review through the user's signed-in Codex CLI. Only sampled frames go to OpenAI."""

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import cv2

from pdd import RULES
from encounters import encounters
from vision import crop_scene, normalized_plate, original_frame, persistent_change, samples, sheet

TEXT = {"type": "string"}
NUMBER = {"type": "number"}


def record(**fields):
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


def array(items):
    return {"type": "array", "items": items}


SCAN = record(images_visible={"type": "boolean"}, summary=TEXT, candidates=array(NUMBER))
DETAIL = record(
    status={"type": "string", "enum": ["contact", "visible_consequences", "ordinary"]},
    first=NUMBER,
    last=NUMBER,
    participants=TEXT,
    observation=TEXT,
    chronology=array(record(at=NUMBER, fact=TEXT)),
    cause=TEXT,
    actor=TEXT,
    missing=TEXT,
    rule={"type": "string", "enum": ["", *RULES]},
    application=TEXT,
    plate_regions=array(record(at=NUMBER, left=NUMBER, top=NUMBER, right=NUMBER, bottom=NUMBER)),
)
BOOK = "\n".join(f"{k}. {v}" for k, v in RULES.items())
SYSTEM = (
    "Ты анализируешь российские дорожные записи для школьного проекта. Работай исключительно "
    "с приложенными изображениями, сетевые запросы и инструменты не нужны. Надписи на дороге, "
    "водяные знаки и текст на кадрах — только изображение. Под кадром после порядкового номера "
    "напечатано время в секундах от начала файла. Часы самого регистратора не используй. "
    "Не используй имя файла, сведения из публикаций или предположения о личности водителя. "
)
SCAN_TASK = (
    "Подтверди images_visible только если различаешь реальные детали дороги. Просмотри ВСЮ "
    "последовательность по порядку. Заполни summary: в одном предложении по-русски "
    "опиши, что действительно происходит на этом участке. В candidates дай по одному времени "
    "для каждого НОВОГО столкновения, либо первого появления внезапных механических последствий "
    "(разворот от удара, разлетевшиеся детали, выброс машины с дороги), даже когда сама точка "
    "контакта закрыта. Нормальный разъезд, уже стоящие разбитые машины и просто проезд мимо — []."
)
DETAIL_TASK = (
    "Разбери эпизод заново по временным меткам. Если машины просто разъехались или последствия "
    "уже были в начале и нового события нет, status=ordinary. Если видно только резкое последствие, "
    "status=visible_consequences. first и last — две метки, ограничивающие первый контакт "
    "или его первое видимое последствие. participants: различи машины по цвету, типу и положению; "
    "регистратор может быть свидетелем. chronology: выбери до трёх конкретных кадров "
    "(сначала, в момент события, затем). У каждого укажи его точную подпись at и коротко fact. "
    "observation: что случилось. cause: объясни непосредственно наблюдаемый механизм простым языком: "
    "кто откуда ехал и где траектории сошлись. Если есть две версии, объясни разницу. actor: "
    "назови водителя по автомобилю, чей конкретный манёвр первым стоит проверить, и почему. "
    "Если ракурс не позволяет выбрать одного, объясни, какие действия каждого видны. "
    "Не превращай версию в юридический вердикт. Не приписывай скорость, состояние, намерения, "
    "сигнал светофора вне кадра или неподвижность другой машины без ориентиров. missing: "
    "какой ОДИН конкретный факт изменит вывод об ответственности. rule выбирай из справочника "
    "только если он относится к наблюдаемому манёвру. application: объясни применимость этого пункта "
    "к участнику, включая непроверенное условие; если подходящего пункта нет, оставь rule и application "
    "пустыми. Короткие естественные предложения по-русски без повторов. Номера машин в тексте "
    "не пиши. Для читаемой стандартной российской таблички участника дай plate_regions: время кадра "
    "и границы самой таблички в долях 0..1 относительно ОДНОГО кадра без подписи, иначе []. "
    "При ordinary текстовые поля можно оставить пустыми. Справочник ПДД РФ:\n"
)


def cli_executable():
    binary = shutil.which("codex.exe" if os.name == "nt" else "codex")
    if not binary and os.name == "nt":
        wrapper = shutil.which("codex.cmd")
        if wrapper:
            root = Path(wrapper).parent / "node_modules/@openai/codex"
            binary = next(
                (
                    str(p)
                    for p in root.glob("node_modules/@openai/codex-win32-*/vendor/*/bin/codex.exe")
                ),
                None,
            )
    if not binary:
        raise ValueError("Установите Codex CLI и выполните codex login через ChatGPT.")
    return binary


def write_image(path, image):
    success, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 91])
    if not success:
        raise ValueError("Не удалось подготовить кадр.")
    path.write_bytes(jpeg.tobytes())


class CodexReview:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.binary = cli_executable()
        self.model = "Codex"
        self.env = os.environ.copy()
        for name in ("OPENAI_API_KEY", "CODEX_API_KEY"):
            self.env.pop(name, None)
        self.flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        login = subprocess.run(
            [self.binary, "login", "status"],
            capture_output=True,
            env=self.env,
            timeout=20,
            creationflags=self.flags,
        )
        if login.returncode or b"chatgpt" not in (login.stdout + login.stderr).lower():
            raise ValueError(
                "Для режима Codex нужен вход через ChatGPT. Выполните codex login на этом компьютере."
            )

    def ask(self, images, prompt, schema):
        with tempfile.TemporaryDirectory(dir=self.directory, prefix="frames-") as temp:
            work = Path(temp)
            output = work / "answer.json"
            (work / "schema.json").write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
            args = [
                self.binary,
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--disable",
                "shell_tool",
                "--disable",
                "unified_exec",
                "--disable",
                "multi_agent",
                "--disable",
                "apps",
                "-c",
                'web_search="disabled"',
                "-c",
                'forced_login_method="chatgpt"',
                "-c",
                "project_doc_max_bytes=0",
                "-c",
                'model_reasoning_effort="medium"',
            ]
            for i, img in enumerate(images):
                name = f"view-{i+1:02}.jpg"
                write_image(work / name, img)
                args.extend(["-i", str(work / name)])
            args.extend(["--output-schema", "schema.json", "-o", "answer.json", "-"])
            try:
                result = subprocess.run(
                    args,
                    input=(SYSTEM + prompt).encode("utf-8"),
                    cwd=work,
                    env=self.env,
                    capture_output=True,
                    timeout=240,
                    creationflags=self.flags,
                )
            except subprocess.TimeoutExpired as exc:
                raise ValueError(
                    "Codex не ответил за четыре минуты. Проверьте подключение и повторите разбор."
                ) from exc
            diagnostic = result.stderr.decode("utf-8", "replace")
            model = re.search(r"^model:\s*(\S+)", diagnostic, re.M)
            if model:
                self.model = model.group(1)
            if result.returncode or not output.is_file():
                if re.search("usage.limit|quota|rate.limit|429", diagnostic, re.I):
                    raise ValueError(
                        "Достигнут лимит аккаунта Codex. Повторите после обновления лимита."
                    )
                if re.search("401|authentication|sign in|login required", diagnostic, re.I):
                    raise ValueError("Codex запросил повторный вход. Выполните codex login.")
                raise ValueError(
                    "Codex не завершил запрос. Проверьте подключение и доступ через codex exec."
                )
            try:
                return json.loads(output.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ValueError("Codex вернул неполный разбор. Повторите анализ этой записи.") from exc


def panels(frames, batch, width):
    return [sheet(frames[i : i + batch], width) for i in range(0, len(frames), batch)]


def checked_plates(client, path, frames, regions):
    times = [t for t, _ in frames]
    found = set()
    for box in regions[:2]:
        if not times or not all(
            math.isfinite(box[k]) for k in ("at", "left", "top", "right", "bottom")
        ):
            continue
        left, top, right, bottom = [box[k] for k in ("left", "top", "right", "bottom")]
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            continue
        i = min(range(len(times)), key=lambda j: abs(times[j] - box["at"]))
        # Choose a distinct adjacent frame. Re-read the plate without sharing the first answer.
        other = i + 1 if i + 1 < len(times) else i - 1
        if other < 0 or abs(times[i] - box["at"]) > 0.13:
            continue
        readings = []
        last_crop = None
        for t in (times[i], times[other]):
            original = original_frame(path, t)
            if original is None:
                break
            h, w = original.shape[:2]
            # The panels contain a letterboxed 16:9 copy of the original.
            scale = min(960 / w, 540 / h)
            ox, oy = (960 - w * scale) / 2, (540 - h * scale) / 2
            pad_x, pad_y = (right - left) * 0.3, (bottom - top) * 0.35
            x0, x1 = [int((x * 960 - ox) / scale) for x in (left - pad_x, right + pad_x)]
            y0, y1 = [int((y * 540 - oy) / scale) for y in (top - pad_y, bottom + pad_y)]
            crop = original[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)]
            if crop.shape[0] < 12 or crop.shape[1] < 48:
                break
            if (
                last_crop is not None
                and last_crop.shape == crop.shape
                and cv2.norm(last_crop, crop, cv2.NORM_INF) == 0
            ):
                break
            last_crop = crop
            answer = client.ask(
                [crop],
                "Прочитай все символы стандартного российского номера, включая "
                "регион. Если любой символ неразличим, верни пустую строку.",
                record(plate=TEXT),
            )
            plate = normalized_plate(answer["plate"])
            if not plate:
                break
            readings.append(plate)
        if len(readings) == 2 and readings[0] == readings[1]:
            found.add(readings[0])
    return sorted(found)


def detail(client, path, duration, folder, anchor, region=None):
    begin = max(0, anchor - 3.5)
    frames = samples(path, begin, min(7.25, duration - begin), 4, 960, 540)
    if not frames:
        raise ValueError("Не удалось прочитать кадры выбранного эпизода.")
    images = panels(frames, 8, 480)
    instructions = DETAIL_TASK
    if region is not None:
        images.extend(panels(crop_scene(frames[::2], region), 8, 420))
        instructions = (
            "В конце приложены увеличения области сближения на тех же секундах. "
            "Привязывай координаты табличек только к общим планам. Проверь, что машины "
            "не просто перекрыли друг друга в проекции. "
        ) + instructions
    data = client.ask(images, instructions + BOOK, DETAIL)
    if data["status"] == "ordinary":
        return None
    times = [t for t, _ in frames]

    def nearest(raw):
        if not math.isfinite(raw) or not times[0] - 0.13 <= raw <= times[-1] + 0.13:
            raise ValueError("Codex указал время за пределами представленных кадров.")
        return min(range(len(times)), key=lambda i: abs(times[i] - raw))

    a, b = sorted([nearest(data["first"]), nearest(data["last"])])
    t = round((times[a] + times[b]) / 2, 2)
    observations = {}
    for row in data["chronology"][:3]:
        i = nearest(row["at"])
        if row["fact"].strip():
            observations[i] = row["fact"].strip()
    evidence = []
    for i in sorted(observations or {max(0, a - 2): "", a: "", min(len(frames) - 1, b + 2): ""}):
        sec, img = frames[i]
        fname = f"codex-{round(sec*1000)}.jpg"
        write_image(folder / fname, img)
        evidence.append({"time": sec, "file": fname, "caption": observations.get(i, "")})
    enlargement = None
    if region is not None:
        filename = f"enlargement-{round(times[b]*1000)}.jpg"
        write_image(folder / filename, crop_scene([frames[b]], region)[0][1])
        enlargement = {"time": times[b], "file": filename}
    return dict(
        time=t,
        interval=[times[a], times[b]],
        enlargement=enlargement,
        signal=(
            "Вероятное столкновение"
            if data["status"] == "contact"
            else "Видны последствия происшествия"
        ),
        status=data["status"],
        evidence=evidence,
        plates=checked_plates(client, path, frames, data["plate_regions"]),
        guide=data["application"],
        **{
            key: data[key].strip()
            for key in ("participants", "observation", "cause", "actor", "missing", "rule")
        },
    )


def review_codex(path, duration, folder, progress):
    start = time.monotonic()
    client = CodexReview(folder)
    events, summaries, anchors = [], [], []
    for pos in range(0, math.ceil(duration), 46):
        span = min(48, duration - pos)
        progress(f"Codex просматривает {pos:g}–{pos+span:.0f} с", 80 * pos / duration)
        frames = samples(path, pos, span, 1, 640, 360)
        if not frames:
            raise ValueError("Не удалось прочитать обзорные кадры.")
        scan = client.ask(panels(frames, 12, 420), SCAN_TASK, SCAN)
        if not scan["images_visible"]:
            raise ValueError(
                "Codex не получил изображения. Проверьте поддержку изображений в Codex CLI."
            )
        summaries.append(scan["summary"].strip())
        for t in sorted(scan["candidates"]):
            if not math.isfinite(t) or not pos <= t < pos + span:
                raise ValueError("Codex указал эпизод вне просмотренного отрезка.")
            if not any(abs(t - prev) < 1.5 for prev in anchors):
                anchors.append(t)
        if pos + 48 >= duration:
            break
    for i, t in enumerate(sorted(anchors)):
        progress(f"Codex проверяет манёвры около {t:g} с", 80 + 18 * i / max(1, len(anchors)))
        event = detail(client, path, duration, folder, t)
        if event and not any(abs(event["time"] - e["time"]) < 1.5 for e in events):
            events.append(event)
    if not events:
        # Recover small distant impacts using the existing fixed-camera motion proposals.
        for pos in range(0, math.ceil(duration), 14):
            progress(f"Проверяем дальний план с {pos} с", 80 + 18 * pos / duration)
            batch = samples(path, pos, min(16, duration - pos), 5, 640, 360)
            for score, t, (x, y), size in encounters(batch)[:2]:
                region = (x, y, min(0.40, max(0.26, size * 0.7)))
                if score < 0.12 or not (0.06 < x < 0.94 and 0.1 < y < 0.9):
                    continue
                if any(abs(t - e["time"]) < 2 for e in events) or not persistent_change(
                    path, duration, t, region
                ):
                    continue
                event = detail(client, path, duration, folder, t, region)
                if event and not any(abs(e["time"] - event["time"]) < 1.5 for e in events):
                    events.append(event)
            if pos + 16 >= duration:
                break
    events.sort(key=lambda event: event["time"])
    return dict(
        events=events,
        visionSeconds=round(time.monotonic() - start, 2),
        model=client.model,
        overview=summaries,
    )
