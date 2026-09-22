param([int]$JevProcessId = 0)
$ErrorActionPreference = 'Stop'
if (-not $JevProcessId) {
    $status = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 3
    $JevProcessId = [int]$status.pid
}
$adapters = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | ForEach-Object {
    [pscustomobject]@{ adapter = $_.Name; dedicated_mib = [math]::Round($_.DedicatedUsage / 1MB, 1); shared_mib = [math]::Round($_.SharedUsage / 1MB, 1) }
})
$processes = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object { $_.Name -like "pid_${JevProcessId}_*" } | ForEach-Object {
    [pscustomobject]@{ adapter = ($_.Name -replace '^pid_\d+_', ''); dedicated_mib = [math]::Round($_.DedicatedUsage / 1MB, 1); shared_mib = [math]::Round($_.SharedUsage / 1MB, 1) }
})
[pscustomobject]@{ at = (Get-Date -Format o); process_id = $JevProcessId; adapters = $adapters; jev_process = $processes } | ConvertTo-Json -Depth 5
