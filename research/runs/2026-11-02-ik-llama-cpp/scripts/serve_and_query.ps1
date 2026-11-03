# Start a llama-server pinned to a CPU mask (GPUs hidden), wait for /health, run the greedy client, report peak memory, stop it.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File serve_and_query.ps1 -Exe X -Mask 0x5555 -Port 8091 -OutPrefix P -ArgLine "..."
param(
  [Parameter(Mandatory=$true)] [string]$Exe,
  [Parameter(Mandatory=$true)] [string]$Mask,
  [Parameter(Mandatory=$true)] [int]$Port,
  [Parameter(Mandatory=$true)] [string]$OutPrefix,
  [Parameter(Mandatory=$true)] [string]$ArgLine
)
$env:CUDA_VISIBLE_DEVICES = "-1"
$client = Join-Path $PSScriptRoot "greedy_client.py"
$p = Start-Process -FilePath $Exe -ArgumentList "$ArgLine --host 127.0.0.1 --port $Port" -NoNewWindow -PassThru -RedirectStandardOutput "$OutPrefix.out" -RedirectStandardError "$OutPrefix.log"
$null = $p.Handle
$p.ProcessorAffinity = [IntPtr][Convert]::ToInt64($Mask, 16)
$ready = $false
$deadline = (Get-Date).AddMinutes(15)
while (-not $ready -and -not $p.HasExited -and (Get-Date) -lt $deadline) {
  Start-Sleep -Seconds 2
  try { $h = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($h.StatusCode -eq 200) { $ready = $true } } catch {}
}
if (-not $ready) { "server not ready (exited=$($p.HasExited))"; if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; exit 1 }
& python -I -X utf8 $client "http://127.0.0.1:$Port" "$OutPrefix.json"
$rc = $LASTEXITCODE
$p.Refresh()
$ws = $p.PeakWorkingSet64; $priv = $p.PeakPagedMemorySize64
Stop-Process -Id $p.Id -Force
"client_exit={0} mask={1} peak_working_set_mib={2:N0} peak_private_mib={3:N0}" -f $rc, $Mask, ($ws / 1MB), ($priv / 1MB)
