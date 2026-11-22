from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether, Table

from pdd import RULES, source


def make_report(records, skipped):
    font_paths = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    font_path = next((p for p in font_paths if p.is_file()), None)
    if font_path is None:
        raise ValueError("Для PDF нужен шрифт Segoe UI, DejaVu Sans или Arial с кириллицей.")
    pdfmetrics.registerFont(TTFont("Roadframe", str(font_path)))
    output = BytesIO()
    doc = SimpleDocTemplate(
        output,
        title="Дорожный кадр · Разбор записи",
        author="Дорожный кадр",
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=36,
    )
    body = []

    def paragraph(text, size=11):
        style = ParagraphStyle(
            "report",
            fontName="Roadframe",
            fontSize=size,
            leading=size * 1.4,
            spaceAfter=6,
            textColor=colors.HexColor("#202124"),
        )
        return Paragraph(escape(str(text)), style)

    body.append(paragraph("Разбор записи по ПДД РФ", 23))
    moments = sum(len(job["view"]["events"]) for _, job in records)
    body.append(
        paragraph(
            f"Просмотрено файлов: {len(records)}. Эпизодов для проверки: {moments}. "
            "Время указано от начала каждого файла."
        )
    )
    for name in skipped:
        body.append(paragraph(f"Файл пропущен при обработке: {name}", 10))
    for path, job in records:
        view = job["view"]
        body.extend(
            [
                Spacer(1, 9),
                paragraph(path, 16),
                paragraph(
                    f'Длительность {view["duration"]:.1f} с · разбор моделью {view["visionSeconds"]:.1f} с',
                    10,
                ),
            ]
        )
        if not view["events"]:
            body.append(
                paragraph(
                    "На просмотренных кадрах модель не выделила столкновение. Между кадрами обзора проходит одна секунда, поэтому короткий эпизод мог быть пропущен."
                )
            )
        for event in view["events"]:
            lo, hi = event["interval"]
            body.append(paragraph(f'{event["signal"]} · {lo:g}–{hi:g} с', 14))
            if event["participants"]:
                body.append(paragraph("Участники эпизода. " + event["participants"]))
            body.append(paragraph(event["observation"]))
            body.append(paragraph("Возможная причина", 12))
            body.append(paragraph(event["cause"] or "По этим кадрам установить причину не удалось."))
            body.append(paragraph("Чьи действия стоит проверить", 12))
            body.append(
                paragraph(
                    event["actor"] or "По этой съёмке нельзя уверенно выделить одного из участников."
                )
            )
            if event["rule"]:
                rule = event["rule"]
                body.append(paragraph(f"ПДД РФ, п. {rule}. " + RULES[rule]))
                body.append(
                    Paragraph(
                        f'<a href="{source(rule)}" color="#0057a8">Открыть текст пункта</a>',
                        ParagraphStyle("link", fontName="Roadframe", fontSize=10, spaceAfter=6),
                    )
                )
            if event.get("guide"):
                body.append(paragraph(event["guide"]))
            if event["missing"] and not event.get("guide"):
                body.append(paragraph("Для проверки. " + event["missing"]))
            body.append(
                paragraph(
                    "Номера на кадрах. "
                    + (", ".join(event["plates"]) or "Полное совпадение по двум кадрам не получено.")
                )
            )
            pictures, captions = [], []
            for item in event["evidence"]:
                picture = job["folder"] / item["file"]
                width, height = ImageReader(str(picture)).getSize()
                pictures.append(Image(str(picture), width=155, height=155 * height / width))
                captions.append(paragraph(f'{item["time"]:g} с', 9))
            if pictures:
                body.append(
                    KeepTogether(
                        [Table([pictures, captions], colWidths=[165] * len(pictures)), Spacer(1, 8)]
                    )
                )
            if event.get("enlargement"):
                item = event["enlargement"]
                picture = job["folder"] / item["file"]
                width, height = ImageReader(str(picture)).getSize()
                body.append(
                    KeepTogether(
                        [
                            paragraph(f'Участок сближения крупнее · {item["time"]:g} с', 10),
                            Image(str(picture), width=240, height=240 * height / width, hAlign="LEFT"),
                        ]
                    )
                )
    body.extend(
        [
            Spacer(1, 12),
            paragraph("Как читать этот отчёт", 12),
            paragraph(
                "Версии предложила локальная модель Qwen3-VL-4B. Для юридического вывода "
                "о виновности нужны все доказательства. Справочник ПДД РФ сверён 22.11.2026. "
                "Для прошлых лет проверьте редакцию правил на дату происшествия.",
                9,
            ),
        ]
    )
    doc.build(body)
    output.seek(0)
    return output
