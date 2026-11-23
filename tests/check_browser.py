"""Optional end-to-end check against a running server using Playwright and Edge."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import video_info
from video_frames import samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18065")
    parser.add_argument("--channel", default="msedge")
    parser.add_argument("--screenshots", type=Path)
    options = parser.parse_args()
    videos = ROOT / "test-videos-ru"
    source = videos / "03-chelyabinsk.mp4"
    with tempfile.TemporaryDirectory() as temp, sync_playwright() as pw:
        folder = Path(temp) / "Archive"
        road = folder / "road"
        road.mkdir(parents=True)
        (folder / "broken.mp4").write_bytes(b"broken video")
        shutil.copy(videos / "07-trekhgorny-control.mp4", road / "2.mp4")
        shutil.copy(source, road / "10.mp4")
        browser = pw.chromium.launch(channel=options.channel)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(options.url + "/?mode=motion")
        page.locator("#folder").set_input_files(str(folder))
        page.wait_for_function("""document.querySelector('#progress-box').hidden &&
            document.querySelectorAll('.file-segment').length === 2""", timeout=180000)
        assert page.get_by_role("tab", name="Эпизоды").get_attribute("aria-selected") == "true"
        # The first file is clear. Once the second is analysed its collision opens automatically.
        assert page.locator("#filename").inner_text() == "road/10.mp4"
        page.locator("#files-tab").click()
        assert page.locator(".file-segment strong").all_text_contents() == ["road/2.mp4", "road/10.mp4"]
        assert "broken.mp4" in page.locator("#error-list").text_content()
        page.locator(".file-segment").first.click()
        assert page.locator("#export").is_hidden()
        assert page.locator("#manual").is_visible()
        page.locator("#events-tab").click()
        event = page.locator(".event")
        assert event.count() == 1, event.all_text_contents()
        event.click()
        page.wait_for_function("""video.readyState >= 2 && !video.seeking &&
            video.currentTime >= 8 && video.currentTime <= 10""")
        assert page.locator("#filename").inner_text() == "road/10.mp4"
        moment = page.locator("video").evaluate("(v) => v.currentTime")
        clock = event.locator("time").inner_text()
        minute, second = clock.replace(",", ".").split(":")
        assert abs(int(minute) * 60 + float(second) - moment) < .05, clock
        assert clock in page.locator("#clock").inner_text()
        assert clock in page.locator("#clip-title").inner_text()
        page.locator("#clip-settings summary").click()
        page.locator("#before").fill("1")
        page.locator("#after").fill("2")
        page.locator("#clip-settings summary").click()
        page.locator("#preview-clip").click()
        page.wait_for_function("!video.paused")
        page.wait_for_function("video.paused && Math.abs(video.currentTime - 10.92) < .06", timeout=15000)
        saved_range = page.locator("#range-label").inner_text()
        page.locator("#forward").click()
        assert page.locator("#range-label").inner_text() == saved_range
        assert event.get_attribute("aria-pressed") == "true"
        with page.expect_download() as download:
            page.locator("#export").click()
        clip = Path(temp) / "clip.mp4"
        download.value.save_as(str(clip))
        assert "ДТП" in download.value.suggested_filename
        assert abs(video_info(clip).duration - 3) < .1
        original = samples(source, moment, .25, 4, 320, 180)[0][1].astype(float)
        exported = samples(clip, 1, .25, 4, 320, 180)[0][1].astype(float)
        difference = np.abs(original - exported).mean()
        assert difference < 4, difference
        event.click()
        for width in (1280, 768, 390):
            page.set_viewport_size({"width": width, "height": 900})
            assert page.evaluate("innerWidth === document.documentElement.scrollWidth")
            if options.screenshots:
                options.screenshots.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(options.screenshots / f"browser-{width}.png"), full_page=True)
        assert not errors, errors
        print(f"OK | event {clock} in file | video {moment:.2f}s | clip 3s | pixel difference {difference:.2f}/255")
        browser.close()


if __name__ == "__main__":
    main()
