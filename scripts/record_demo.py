from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs" / "video"
DELIVERABLES = ROOT / "deliverables"
RAW_VIDEO = OUTPUTS / "claimscope-demo.webm"
NARRATION = OUTPUTS / "claimscope-narration.wav"
FINAL_VIDEO = DELIVERABLES / "ClaimScope-demo.mp4"
URL = os.getenv("CLAIMSCOPE_DEMO_URL", "http://localhost:8507")
BROWSER_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]
NARRATION_TEXT = (
    "ClaimScope 不直接生成想法，而是先把模糊研究方向转成可证伪的核心主张。"
    "结果显式展示机制、目标、预期效果，以及缺失的比较基线、评价指标和边界条件。"
    "在大模型模式下，三个提议者、两个对抗评审和一个裁决者交换结构化公开产物；外部服务失败时，系统会安全降级。"
    "完整任务链继续扩展主张变体、拆解隐含假设，并生成支持、反驳、限制和零结果查询。"
    "证据被映射回每项假设，未知与反驳严格分离，合成演示文献也会醒目标记。"
    "负结果和悬而未决的假设最终变成有边界、可执行的实验机会。"
    "仓库提供五个 MCP 工具、九十六项通过测试，以及二十七案例的七十点零九分基线。"
    "这个分数只用于回归检查，不代表科学有效性。"
)


def wait_for_app(page) -> None:
    page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
    page.get_by_text("ClaimScope", exact=True).wait_for(timeout=60_000)
    page.wait_for_timeout(2_000)


def app_is_running() -> bool:
    try:
        return urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2).read().strip() == b"ok"
    except Exception:
        return False


def start_app() -> subprocess.Popen | None:
    if app_is_running():
        return None
    command = [str(Path(os.sys.executable)), "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8507"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
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


def record_browser() -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for old in OUTPUTS.glob("*.webm"):
        old.unlink()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=str(browser_executable()),
            args=["--disable-gpu", "--font-render-hinting=none"],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(OUTPUTS),
            record_video_size={"width": 1280, "height": 720},
        )
        page = context.new_page()
        wait_for_app(page)
        page.get_by_role("button", name="Run Arena").click()
        page.get_by_text("Core Claim", exact=True).wait_for(timeout=30_000)
        page.wait_for_timeout(6_000)
        page.mouse.wheel(0, 520)
        page.wait_for_timeout(4_000)
        page.mouse.wheel(0, 480)
        page.wait_for_timeout(3_000)
        page.get_by_text("Full Discovery", exact=True).first.click()
        page.get_by_role("button", name="Run Discovery").wait_for(timeout=20_000)
        page.get_by_role("button", name="Run Discovery").click()
        page.get_by_text("Testable assumptions", exact=True).wait_for(timeout=30_000)
        page.wait_for_timeout(4_000)
        page.get_by_role("tab", name="Assumption Ledger").click()
        page.wait_for_timeout(5_000)
        page.get_by_role("tab", name="Evidence").click()
        page.wait_for_timeout(8_000)
        page.get_by_role("tab", name="Opportunities").click()
        page.wait_for_timeout(7_000)
        page.get_by_role("tab", name="Trace").click()
        page.wait_for_timeout(6_000)
        page.set_content(
            """
            <html><body style="margin:0;background:#f5f8f8;color:#17212b;font-family:Arial,'Microsoft YaHei',sans-serif;">
            <main style="height:720px;display:grid;place-content:center;text-align:center;">
              <div style="font-size:54px;font-weight:750;color:#1f5a7a;">ClaimScope</div>
              <div style="font-size:26px;margin-top:18px;">Assumption-centric research discovery</div>
              <div style="display:flex;gap:18px;justify-content:center;margin-top:42px;">
                <b style="padding:18px 24px;background:white;border:1px solid #cad7da;">96 tests passed</b>
                <b style="padding:18px 24px;background:white;border:1px solid #cad7da;">27 ClaimBench cases</b>
                <b style="padding:18px 24px;background:white;border:1px solid #cad7da;">5 MCP tools</b>
              </div>
              <div style="margin-top:46px;font-size:20px;color:#53636d;">github.com/yueqian-coder/multi</div>
              <div style="margin-top:12px;font-size:16px;color:#8a5550;">Synthetic fixtures are demo data, not scientific evidence.</div>
            </main></body></html>
            """
        )
        page.wait_for_timeout(6_000)
        video = page.video
        context.close()
        browser.close()
        recorded = Path(video.path())
    recorded.replace(RAW_VIDEO)
    return RAW_VIDEO


def synthesize_narration() -> Path:
    escaped = NARRATION_TEXT.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$names=@($s.GetInstalledVoices() | ForEach-Object {$_.VoiceInfo.Name}); "
        "$voice=$names | Where-Object {$_ -match 'Huihui|Yaoyao|Kangkang'} | Select-Object -First 1; "
        "if(-not $voice){$voice=$names | Select-Object -First 1}; "
        "if(-not $voice){throw 'No Windows speech voice is installed'}; $s.SelectVoice($voice); "
        "$s.Rate=1; $s.Volume=100; "
        f"$s.SetOutputToWaveFile('{NARRATION}'); $s.Speak('{escaped}'); "
        "$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    return NARRATION


def compose_video() -> Path:
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(RAW_VIDEO),
            "-i",
            str(NARRATION),
            "-filter_complex",
            "[0:v]tpad=stop_mode=clone:stop_duration=20[v];[1:a]atempo=1.16,apad=pad_dur=20[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            "58",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(FINAL_VIDEO),
        ],
        check=True,
    )
    return FINAL_VIDEO


if __name__ == "__main__":
    started = time.perf_counter()
    server = start_app()
    try:
        print(record_browser())
        print(synthesize_narration())
        print(compose_video())
        print(f"elapsed={time.perf_counter() - started:.1f}s")
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
