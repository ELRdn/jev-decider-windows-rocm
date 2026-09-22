[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)][string]$PythonPath,
    [Parameter(Mandatory)][string]$DeciderDirectory,
    [string]$StateDirectory = (Join-Path $env:USERPROFILE '.jev-gateway'),
    [string]$NodePath,
    [string]$GatewayLauncher,
    [ValidateSet('int8', 'bf16')][string]$Precision = 'int8',
    [ValidateRange(1, 3600)][int]$IdleSeconds = 10,
    [ValidateRange(2, 7200)][int]$DeepIdleSeconds = 30,
    [switch]$RegisterLogonTask,
    [switch]$StartNow
)
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'This installer requires Windows.' }
if ($DeepIdleSeconds -le $IdleSeconds) { throw 'DeepIdleSeconds must exceed IdleSeconds.' }
$repoRoot = Split-Path $PSScriptRoot -Parent
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$DeciderDirectory = (Resolve-Path -LiteralPath $DeciderDirectory).Path
$pythonWindowless = Join-Path (Split-Path $PythonPath -Parent) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonWindowless -PathType Leaf)) { throw 'pythonw.exe must accompany the supplied Python.' }
if (-not $NodePath) { $NodePath = (Get-Command node -ErrorAction Stop).Source }
$NodePath = (Resolve-Path -LiteralPath $NodePath).Path
if (-not $GatewayLauncher) {
    $npmRoot = (& npm root -g).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'npm root -g failed.' }
    $GatewayLauncher = Join-Path $npmRoot 'jev-gateway\bin\jev-codex.mjs'
}
$GatewayLauncher = (Resolve-Path -LiteralPath $GatewayLauncher).Path
$stateRoot = [IO.Path]::GetFullPath($StateDirectory)
$runtimeRoot = Join-Path $stateRoot 'local'
$managerPath = Join-Path $runtimeRoot 'jev_local.py'
$wrapperPath = Join-Path $runtimeRoot 'jev-local.ps1'
$cacheRoot = Join-Path $env:LOCALAPPDATA 'TritonCache\decider-jev'
$utf8 = [Text.UTF8Encoding]::new($false)

if (-not $PSCmdlet.ShouldProcess($stateRoot, 'Install local Jev service, merge gateway settings, and optionally register/start the logon task')) { return }

# Updating files beneath a running backend would mix old and new implementations.
$running = $null
try { $running = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 2 } catch { }
if ($running) { throw 'Port 8000 is serving a process. Stop the existing local service before installing.' }
$taskName = 'Jev Local Decider'
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask -and $existingTask.State -eq 'Running') { throw 'The existing Jev task is still running. Stop it before updating.' }
if ($RegisterLogonTask -and $existingTask -and -not ($existingTask.Actions.Arguments -like ('*' + $managerPath + '*'))) {
    throw 'The task name is already used by another installation. No task was replaced.'
}

$backupRoot = Join-Path $stateRoot ('backups\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $runtimeRoot, $backupRoot -Force | Out-Null
foreach ($name in @('local_rocm_server.py', 'jev_weight_only.py', 'jev_local.py', 'installation.json', 'jev-local.ps1')) {
    $existing = Join-Path $runtimeRoot $name
    if (Test-Path -LiteralPath $existing) { Copy-Item -LiteralPath $existing -Destination (Join-Path $backupRoot $name) }
}
$envPath = Join-Path $stateRoot '.env'
if (Test-Path -LiteralPath $envPath) { Copy-Item -LiteralPath $envPath -Destination (Join-Path $backupRoot '.env') }
if ($RegisterLogonTask -and $existingTask) {
    [IO.File]::WriteAllText((Join-Path $backupRoot 'task.xml'), (Export-ScheduledTask -TaskName $taskName), $utf8)
}
foreach ($name in @('local_rocm_server.py', 'jev_weight_only.py', 'jev_local.py')) {
    Copy-Item -LiteralPath (Join-Path $repoRoot "service\$name") -Destination (Join-Path $runtimeRoot $name)
}
$config = [ordered]@{
    python = $PythonPath; pythonw = $pythonWindowless; decider_dir = $DeciderDirectory
    node = $NodePath; gateway_launcher = $GatewayLauncher; cache = $cacheRoot
    upstream = 'https://chatgpt.com/backend-api/codex'; precision = $Precision
    idle_seconds = $IdleSeconds; deep_idle_seconds = $DeepIdleSeconds; backup = $backupRoot
}
[IO.File]::WriteAllText((Join-Path $runtimeRoot 'installation.json'), ($config | ConvertTo-Json) + "`n", $utf8)
$template = [IO.File]::ReadAllLines((Join-Path $repoRoot '.env.example'))
$updates = [ordered]@{}
foreach ($line in $template) {
    if ($line -match '^([A-Z0-9_]+)=(.*)$') { $updates[$Matches[1]] = $Matches[2] }
}
$oldLines = @()
if (Test-Path -LiteralPath $envPath) { $oldLines = [IO.File]::ReadAllLines($envPath) }
if ($oldLines | Where-Object { $_ -match '^\s*TYPESAFE_API_KEY\s*=\s*\S+' }) { $updates.Remove('TYPESAFE_API_KEY') }
$newLines = @($oldLines | Where-Object { -not $updates.Contains(($_ -split '=', 2)[0].Trim()) })
$newLines += @($updates.GetEnumerator() | ForEach-Object { $_.Key + '=' + $_.Value })
[IO.File]::WriteAllLines($envPath, $newLines, $utf8)
$escapedPython = $PythonPath.Replace("'", "''")
$escapedManager = $managerPath.Replace("'", "''")
[IO.File]::WriteAllText($wrapperPath, "& '$escapedPython' '$escapedManager' @args`nexit `$LASTEXITCODE`n", $utf8)

if ($RegisterLogonTask) {
    $taskUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $action = New-ScheduledTaskAction -Execute $pythonWindowless -Argument ('"' + $managerPath + '" run') -WorkingDirectory $runtimeRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
    $principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
}
if ($StartNow) { & $PythonPath $managerPath start; if ($LASTEXITCODE -ne 0) { throw 'Service start failed.' } }
Write-Output "Installed: $wrapperPath"
Write-Output "Backup: $backupRoot"
Write-Output 'Merge the Codex provider configuration described in README.md. Restart an already-running gateway to reload its environment.'
