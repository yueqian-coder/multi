$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CodexRoot = Join-Path $RepoRoot ".codex"
$SkillsRoot = Join-Path $CodexRoot "skills"
$MiniMaxRoot = Join-Path $CodexRoot "minimax-skills"
$MiniMaxCommit = "60aaae52bb2af8162732751a4332f62a5fef518b"
$UiProVersion = "2.10.2"

New-Item -ItemType Directory -Force -Path $SkillsRoot | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $MiniMaxRoot ".git"))) {
    git clone https://github.com/MiniMax-AI/skills.git $MiniMaxRoot
}
git -C $MiniMaxRoot fetch origin $MiniMaxCommit --depth 1
git -C $MiniMaxRoot checkout --detach $MiniMaxCommit

$MiniMaxLink = Join-Path $SkillsRoot "minimax-skills"
if (-not (Test-Path -LiteralPath $MiniMaxLink)) {
    cmd /c mklink /J $MiniMaxLink (Join-Path $MiniMaxRoot "skills") | Out-Null
}

$RuntimeRoot = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies"
$RuntimeNode = Join-Path $RuntimeRoot "node\bin"
$RuntimeFallback = Join-Path $RuntimeRoot "bin\fallback"
if (Test-Path -LiteralPath (Join-Path $RuntimeNode "node.exe")) {
    $env:PATH = "$RuntimeNode;$RuntimeFallback;$env:PATH"
}
$Pnpm = Get-Command pnpm.cmd -ErrorAction SilentlyContinue
$Npx = Get-Command npx.cmd -ErrorAction SilentlyContinue
if ($Pnpm) {
    & $Pnpm.Source dlx "ui-ux-pro-max-cli@$UiProVersion" init --ai codex --offline --force
} elseif ($Npx) {
    & $Npx.Source -y "ui-ux-pro-max-cli@$UiProVersion" init --ai codex --offline --force
} else {
    throw "Install Node.js with pnpm or npm before installing UI/UX Pro Max."
}

if (-not (Test-Path -LiteralPath (Join-Path $SkillsRoot "ui-ux-pro-max\SKILL.md"))) {
    throw "UI/UX Pro Max installation did not create its Codex skill."
}

Write-Output "MiniMax skills: $MiniMaxCommit"
Write-Output "UI/UX Pro Max: $UiProVersion"
Write-Output "Restart Codex to refresh project skill discovery."
