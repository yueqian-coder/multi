from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.getenv("CLAIMSCOPE_UI_ASSETS", ROOT / "docs" / "assets"))
SCREENSHOT_PREFIX = os.getenv("CLAIMSCOPE_UI_PREFIX", "claimscope-v07")
URL = os.getenv("CLAIMSCOPE_UI_URL", "http://localhost:8511")
PARSED_URL = urlparse(URL)
PORT = PARSED_URL.port or (443 if PARSED_URL.scheme == "https" else 80)
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
    process_env = os.environ.copy()
    process_env.setdefault("CLAIMSCOPE_PROVIDER", "gpt")
    process_env.setdefault("CLAIMSCOPE_GPT_API_KEY", "ui-audit-placeholder")
    process_env.setdefault("CLAIMSCOPE_GPT_MODEL", "gpt-ui-audit")
    process_env.setdefault("OPENAI_BASE_URL", "https://provider.invalid/v1")
    process_env.setdefault("CLAIMSCOPE_TRUST_ENV_PROVIDER", "1")
    process_env.setdefault(
        "CLAIMSCOPE_RESULT_SNAPSHOT", str(ROOT / "examples" / "core_claim_result.json")
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless",
            "true",
            "--server.port",
            str(PORT),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=process_env,
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


def stabilize_frame(page) -> None:
    page.wait_for_timeout(2_000)
    page.evaluate(
        """() => new Promise(resolve => requestAnimationFrame(
            () => requestAnimationFrame(resolve)
        ))"""
    )
    page.screenshot(animations="disabled")
    page.wait_for_timeout(250)


def audit_page(page, *, mobile: bool = False, require_result: bool = False) -> None:
    body_text = page.locator("body").inner_text()
    for marker in ("Traceback", "NameError", "SyntaxError", "Exception"):
        if marker in body_text:
            raise AssertionError(f"UI contains runtime error marker: {marker}")
    dimensions = page.evaluate(
        """() => ({
            viewport: window.innerWidth,
            document: document.documentElement.scrollWidth,
        })"""
    )
    if dimensions["document"] > dimensions["viewport"]:
        raise AssertionError(f"horizontal overflow detected: {dimensions}")

    result = page.locator(".result-header")
    rail = page.locator(".run-rail")
    if require_result and (result.count() != 1 or rail.count() != 1):
        raise AssertionError("result fixture and run rail are required for UI auditing")
    if result.count() and rail.count():
        result_box = result.bounding_box()
        rail_box = rail.bounding_box()
        if result_box is None or rail_box is None:
            raise AssertionError("result or run rail is not visible")
        if mobile and result_box["y"] >= rail_box["y"]:
            raise AssertionError("mobile result must appear before the run rail")
        if not mobile and rail_box["x"] >= result_box["x"]:
            raise AssertionError("desktop run rail must appear left of the result")


def audit_provider_profiles(page) -> None:
    page.get_by_role("button", name="模型服务", exact=True).click()
    popover = page.locator('[data-testid="stPopoverBody"]')
    popover.wait_for(state="visible", timeout=10_000)
    for label in ("GPT", "Claude", "自定义"):
        radio = popover.get_by_role("radio", name=label, exact=True)
        if radio.count():
            radio.wait_for(timeout=10_000)
        else:
            popover.get_by_text(label, exact=True).wait_for(timeout=10_000)
    page.keyboard.press("Escape")


def audit_result_views(page) -> None:
    if page.locator(".result-header").count() != 1:
        raise AssertionError("core-claim result fixture is required for UI auditing")
    for label in ("结论概览", "候选与审查", "角色输出"):
        tab = page.get_by_role("tab", name=label)
        tab.click()
        tab.wait_for(state="visible", timeout=10_000)
        if tab.get_attribute("aria-selected") != "true":
            raise AssertionError(f"result tab did not become selected: {label}")
        panel = page.locator('[role="tabpanel"]:visible')
        if not panel.count() or not panel.first.inner_text().strip():
            raise AssertionError(f"result tab has no visible content: {label}")
    page.get_by_role("tab", name="结论概览").click()


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
                audit_page(page, mobile=name == "mobile", require_result=True)
                if name == "desktop":
                    audit_provider_profiles(page)
                    audit_result_views(page)
                stabilize_frame(page)
                page.screenshot(
                    path=ASSETS / f"{SCREENSHOT_PREFIX}-{name}.png",
                    full_page=False,
                    animations="disabled",
                )
                context.close()

            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            wait_for_ready(page)
            page.get_by_text("EN", exact=True).click()
            page.get_by_text("Research direction", exact=True).first.wait_for(timeout=30_000)
            stabilize_frame(page)
            audit_page(page, require_result=True)
            page.screenshot(
                path=ASSETS / f"{SCREENSHOT_PREFIX}-english.png",
                full_page=False,
                animations="disabled",
            )
            context.close()

            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            wait_for_ready(page)
            page.get_by_text("证据发现", exact=True).first.click()
            page.get_by_text("最多检索论文数", exact=True).wait_for(timeout=30_000)
            stabilize_frame(page)
            audit_page(page)
            page.screenshot(
                path=ASSETS / f"{SCREENSHOT_PREFIX}-discovery.png",
                full_page=False,
                animations="disabled",
            )
            context.close()
            browser.close()
    finally:
        if server is not None and server.poll() is None:
            server.terminate()


if __name__ == "__main__":
    main()
