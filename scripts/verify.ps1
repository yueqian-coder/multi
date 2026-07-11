$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv (Join-Path $RepoRoot ".venv")
    & $Python -m pip install -e "$RepoRoot[web,mcp,dev]" build
}
& $Python -m pip install -r scripts\demo-requirements.txt | Out-Null

& $Python -m pytest -q
& $Python -m compileall -q claimscope app.py
& $Python -m py_compile scripts\build_course_report.py scripts\record_demo.py scripts\capture_ui.py
& $Python -m claimscope.cli benchmark --engine heuristic --output outputs\claimbench-final.json
& $Python -m build --wheel

$Wheel = Get-ChildItem (Join-Path $RepoRoot "dist\claimscope-*.whl") | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Smoke = Join-Path $RepoRoot ("outputs\verify-smoke-" + [Guid]::NewGuid().ToString("N"))
& $Python -m venv $Smoke
$SmokePython = Join-Path $Smoke "Scripts\python.exe"
& $SmokePython -m pip install "$($Wheel.FullName)[web,mcp]" | Out-Null
Push-Location $Smoke
try {
    & (Join-Path $Smoke "Scripts\claimscope.exe") benchmark --engine heuristic | Out-Null
    & (Join-Path $Smoke "Scripts\claimscope-mcp.exe") --help | Out-Null
} finally {
    Pop-Location
}

$StartedServer = $null
try {
    try {
        $Health = (Invoke-WebRequest -UseBasicParsing http://localhost:8507/_stcore/health -TimeoutSec 2).Content
    } catch {
        $StartedServer = Start-Process -FilePath $SmokePython -ArgumentList "-m","streamlit","run","app.py","--server.headless","true","--server.port","8507" -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            try {
                $Health = (Invoke-WebRequest -UseBasicParsing http://localhost:8507/_stcore/health -TimeoutSec 2).Content
                if ($Health -eq "ok") { break }
            } catch {}
            Start-Sleep -Seconds 1
        }
    }
    if ($Health -ne "ok") { throw "Streamlit health check failed." }
} finally {
    if ($StartedServer -and -not $StartedServer.HasExited) { Stop-Process -Id $StartedServer.Id }
}

foreach ($artifact in "ClaimScope-course-report.docx","ClaimScope-course-report.pdf","ClaimScope-demo.mp4") {
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "deliverables\$artifact"))) {
        throw "Missing review artifact: $artifact"
    }
}

& $Python -c "import imageio_ffmpeg,pathlib; p=pathlib.Path('deliverables/ClaimScope-demo.mp4'); frames,seconds=imageio_ffmpeg.count_frames_and_secs(p); assert 0 < seconds < 60, seconds; print(f'VIDEO {frames} frames {seconds:.1f}s')"

$secretPatterns = @(
    'sk-[A-Za-z0-9_-]{32,}',
    'gh[pousr]_[A-Za-z0-9]{36,}',
    'github_pat_[A-Za-z0-9_]{50,}'
)
$tracked = git ls-files
foreach ($file in $tracked) {
    if ([IO.Path]::GetExtension($file) -in ".png", ".docx", ".pdf", ".mp4") { continue }
    $text = Get-Content -Raw -LiteralPath $file -ErrorAction SilentlyContinue
    foreach ($pattern in $secretPatterns) {
        if ($text -match $pattern) { throw "Potential secret in $file" }
    }
}

Write-Output "ClaimScope verification complete."
