from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

from record_demo import URL, browser_executable, start_app


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def wait_for_ready(page) -> None:
    page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
    page.get_by_text("ClaimScope", exact=True).wait_for(timeout=60_000)
    page.wait_for_timeout(1_500)


def run_arena(page) -> None:
    page.get_by_role("button", name="Run Arena").click()
    page.get_by_text("Core Claim", exact=True).wait_for(timeout=30_000)
    page.wait_for_timeout(1_000)


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
                run_arena(page)
                page.screenshot(path=ASSETS / f"claimscope-v03-{name}.png", full_page=False)
                context.close()

            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            wait_for_ready(page)
            page.get_by_text("Full Discovery", exact=True).first.click()
            page.get_by_role("button", name="Run Discovery").click()
            page.get_by_text("Testable assumptions", exact=True).wait_for(timeout=30_000)
            page.get_by_role("tab", name="Evidence").click()
            page.wait_for_timeout(1_000)
            page.screenshot(path=ASSETS / "claimscope-v03-discovery.png", full_page=False)
            context.close()
            browser.close()
    finally:
        if server is not None and server.poll() is None:
            server.terminate()


if __name__ == "__main__":
    main()
