param([string]$TestRoot = (Join-Path ([IO.Path]::GetTempPath()) ('jev-install-test-' + [guid]::NewGuid().ToString('N'))))
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$fixtureRoot = [IO.Path]::GetFullPath($TestRoot)
if (Test-Path -LiteralPath $fixtureRoot) { throw 'Use a new, empty fixture path.' }
$fakeEnvironment = Join-Path $fixtureRoot "environment with space's"
$fakeDecider = Join-Path $fixtureRoot 'decider'
$stateRoot = Join-Path $fixtureRoot 'state'
New-Item -ItemType Directory -Path $fakeEnvironment, $fakeDecider, $stateRoot -Force | Out-Null
foreach ($name in @('python.exe', 'pythonw.exe', 'node.exe', 'jev-codex.mjs')) {
    [IO.File]::WriteAllText((Join-Path $fakeEnvironment $name), 'fixture-only: not executable')
}
[IO.File]::WriteAllText((Join-Path $stateRoot '.env'), "TYPESAFE_API_KEY=synthetic-existing-key`nUNRELATED_SETTING=keep-me`nJEV_MODEL=old-placeholder`n")
# Read-only system inspection is mocked. No process, task, auth or real gateway is changed.
function Invoke-RestMethod { param($Uri, $TimeoutSec) throw 'Fixture: no backend running' }
function Get-ScheduledTask { param($TaskName, $ErrorAction) return $null }
$argsToInstall = @{
    PythonPath = (Join-Path $fakeEnvironment 'python.exe'); DeciderDirectory = $fakeDecider
    NodePath = (Join-Path $fakeEnvironment 'node.exe'); GatewayLauncher = (Join-Path $fakeEnvironment 'jev-codex.mjs')
    StateDirectory = $stateRoot
}
& (Join-Path $repoRoot 'scripts\Install.ps1') @argsToInstall -WhatIf
if (Test-Path -LiteralPath (Join-Path $stateRoot 'local')) { throw 'WhatIf modified the target.' }
& (Join-Path $repoRoot 'scripts\Install.ps1') @argsToInstall
$runtimeRoot = Join-Path $stateRoot 'local'
$config = Get-Content -LiteralPath (Join-Path $runtimeRoot 'installation.json') -Raw | ConvertFrom-Json
if ($config.python -ne $argsToInstall.PythonPath -or $config.precision -ne 'int8' -or $config.deep_idle_seconds -ne 30) { throw 'Generated configuration is incorrect.' }
$envText = Get-Content -LiteralPath (Join-Path $stateRoot '.env') -Raw
foreach ($expected in @('TYPESAFE_API_KEY=synthetic-existing-key', 'UNRELATED_SETTING=keep-me', 'JEV_MODEL=decider-2b')) {
    if (-not $envText.Contains($expected)) { throw 'Environment merge lost a required value.' }
}
if ($envText.Contains('old-placeholder') -or $envText.Contains('TYPESAFE_API_KEY=local-placeholder')) { throw 'Environment merge retained or overwrote the wrong key.' }
if (-not (Test-Path -LiteralPath (Join-Path $config.backup '.env'))) { throw 'Previous environment was not backed up.' }
foreach ($name in @('local_rocm_server.py', 'jev_weight_only.py', 'jev_local.py')) {
    if ((Get-FileHash -LiteralPath (Join-Path $runtimeRoot $name)).Hash -ne (Get-FileHash -LiteralPath (Join-Path $repoRoot "service\$name")).Hash) { throw 'Deployed source differs.' }
}
$parseTokens = $null
$parseErrors = $null
[System.Management.Automation.Language.Parser]::ParseFile((Join-Path $runtimeRoot 'jev-local.ps1'), [ref]$parseTokens, [ref]$parseErrors) | Out-Null
if ($parseErrors.Count) { throw 'Wrapper quoting failed for spaces/apostrophes.' }
Write-Output 'PASS: preview is read-only; source, config, backup, env merge and wrapper quoting are correct.'
Write-Output "Fixture retained for inspection: $fixtureRoot"
