"""Measure a real folder import in Edge, including upload, search and playable preview."""
import argparse
import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def score(results, labels):
    if set(results) != set(labels):
        raise ValueError("Files in the folder and annotation manifest must match.")
    hits = misses = false_marks = 0
    for name, intervals in labels.items():
        remaining = sorted(e["time"] for e in results[name]["events"])
        for low, high in sorted(intervals):
            candidates = [t for t in remaining if low - 1 <= t <= high + 1]
            if candidates:
                remaining.remove(min(candidates, key=lambda t: abs(t - (low + high) / 2)))
                hits += 1
            else:
                misses += 1
        false_marks += len(remaining)
    return dict(found=hits, missed=misses, false_marks=false_marks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18065")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--labels", type=Path, help="Reference intervals in seconds, keyed by filename")
    args = parser.parse_args()
    folder = args.folder.resolve()
    count = sum(p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".avi"} for p in folder.rglob("*"))
    if not count:
        parser.error("The folder contains no supported video files.")
    ready_jobs = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge")
        page = browser.new_page()

        def response(res):
            if re.search(r"/api/jobs/[0-9a-f]+$", res.url) and res.status == 200:
                item = res.json()
                if item.get("status") == "ready":
                    ready_jobs[item["id"]] = {k: item[k] for k in ("duration", "events", "scan") if k in item}

        page.goto(args.url + "/?mode=motion")
        page.on("response", response)
        started = time.monotonic()
        page.locator("#folder").set_input_files(str(folder))
        page.wait_for_function("""n => document.querySelector('#progress-box').hidden &&
            (document.querySelectorAll('.file-segment').length === n ||
             !document.querySelector('#file-errors').hidden)""", arg=count, timeout=300000)
        assert page.locator("#file-errors").is_hidden(), page.locator("#error-list").text_content()
        page.wait_for_function("""() => {
            const video = document.querySelector('video');
            const time = document.querySelector('.event[aria-pressed="true"] time')?.textContent;
            const seconds = time?.replace(',', '.').split(':').reduce((sum, n) => 60 * sum + Number(n), 0) || 0;
            return video.readyState >= 2 && !video.seeking && Math.abs(video.currentTime - seconds) < .25;
        }""", timeout=90000)
        elapsed = round(time.monotonic() - started, 2)
        paths = page.locator(".file-segment strong").all_text_contents()
        assert len(ready_jobs) == len(paths) == count
        results = dict(zip(paths, ready_jobs.values()))
        if args.screenshot:
            page.screenshot(path=str(args.screenshot), full_page=True)
        for event in page.locator(".event").all():
            digits = event.locator("time").inner_text().replace(",", ".").split(":")
            position = sum(float(v) * 60 ** i for i, v in enumerate(reversed(digits)))
            event.click()
            page.wait_for_function("""t => video.readyState >= 2 && !video.seeking &&
                Math.abs(video.currentTime - t) < .25""", arg=position, timeout=60000)
        reviewed = round(time.monotonic() - started, 2)
        browser.close()
    report = dict(seconds=elapsed, all_marks_playable_seconds=reviewed, files=results)
    if args.labels:
        report.update(score(results, json.loads(args.labels.read_text(encoding="utf8"))))
    data = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(data, encoding="utf8")
    print(data)


if __name__ == "__main__":
    main()
