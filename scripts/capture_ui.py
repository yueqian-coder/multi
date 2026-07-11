from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
URL = os.getenv("CLAIMSCOPE_UI_URL", "http://localhost:8507")
BROWSER_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def app_is_running() -> bool:
    try:
        return urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2).read().strip() == b"ok"
    except Exception:
        return False


def start_app() -> subprocess.Popen | None:
    if app_is_running():
        return None
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8507"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    for _ in range(40):
        if app_is_running():
            return process
        if process.poll() is not None:
            raise RuntimeError("Streamlit exited before becoming healthy.")
        time.sleep(0.5)
    process.terminate()
    raise RuntimeError("Timed out waiting for Streamlit health check.")


def browser_executable() -> Path:
    try:
        return next(path for path in BROWSER_CANDIDATES if path.exists())
    except StopIteration as error:
        raise RuntimeError("Install Chrome or Edge, or update BROWSER_CANDIDATES.") from error


def wait_for_ready(page) -> None:
    page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
    page.get_by_text("ClaimScope", exact=True).wait_for(timeout=60_000)
    page.wait_for_timeout(1_500)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    server = start_app()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, executable_path=str(browser_executable()))
            for name, width, height in [
                ("desktop", 1440, 900),
                ("tablet", 768, 1024),
                ("mobile", 390, 844),
            ]:
                context = browser.new_context(viewport={"width": width, "height": height})
                page = context.new_page()
                wait_for_ready(page)
                page.screenshot(path=ASSETS / f"claimscope-v04-{name}.png", full_page=False)
                context.close()

            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            wait_for_ready(page)
            page.get_by_text("EN", exact=True).click()
            page.get_by_text("Research direction", exact=True).first.wait_for(timeout=30_000)
            page.screenshot(path=ASSETS / "claimscope-v04-english.png", full_page=False)
            context.close()

            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            wait_for_ready(page)
            page.get_by_text("完整研究发现", exact=True).first.click()
            page.get_by_text("最多检索论文数", exact=True).wait_for(timeout=30_000)
            page.screenshot(path=ASSETS / "claimscope-v04-discovery.png", full_page=False)
            context.close()
            browser.close()
    finally:
        if server is not None and server.poll() is None:
            server.terminate()


if __name__ == "__main__":
    main()
