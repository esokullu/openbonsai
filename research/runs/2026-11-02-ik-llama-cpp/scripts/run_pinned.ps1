# Run one executable pinned to a CPU mask, with the GPUs hidden, and report its peak memory.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File run_pinned.ps1 -Exe X -Mask 0x5555 -Out o.txt -Err e.txt -ArgLine "..." [-EnvSet NAME=VALUE]
param(
  [Parameter(Mandatory=$true)] [string]$Exe,
  [Parameter(Mandatory=$true)] [string]$Mask,
  [Parameter(Mandatory=$true)] [string]$Out,
  [Parameter(Mandatory=$true)] [string]$Err,
  [Parameter(Mandatory=$true)] [string]$ArgLine,
  [string]$EnvSet = ""
)
$env:CUDA_VISIBLE_DEVICES = "-1"
if ($EnvSet) { $kv = $EnvSet -split '=', 2; Set-Item -Path ("env:" + $kv[0]) -Value $kv[1] }
$t0 = Get-Date
$p = Start-Process -FilePath $Exe -ArgumentList $ArgLine -NoNewWindow -PassThru -RedirectStandardOutput $Out -RedirectStandardError $Err
$null = $p.Handle   # keep the handle so ExitCode is readable after exit
$p.ProcessorAffinity = [IntPtr][Convert]::ToInt64($Mask, 16)
$peakWS = 0; $peakPriv = 0
while (-not $p.HasExited) {
  try {
    $p.Refresh()
    $peakWS = [math]::Max($peakWS, $p.PeakWorkingSet64)
    $peakPriv = [math]::Max($peakPriv, $p.PeakPagedMemorySize64)
  } catch {}
  Start-Sleep -Milliseconds 500
}
$p.WaitForExit()
$secs = ((Get-Date) - $t0).TotalSeconds
"exit={0} secs={1:N1} mask={2} peak_working_set_mib={3:N0} peak_private_mib={4:N0}" -f $p.ExitCode, $secs, $Mask, ($peakWS / 1MB), ($peakPriv / 1MB)
