# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0
#
# Restarts the v21 pipeline if it is not running and not finished.
#
# The pipeline itself is resumable -- every slice writes its own curve file and
# a rerun skips the ones already on disk -- so the only thing missing was
# something to notice it had stopped. nohup does not survive a shell being torn
# down, and a background job did get killed that way. A scheduled task does not
# share that fate: it is owned by the task scheduler, not by whatever launched
# it.
#
# Register (no admin needed, runs as the logged-in user):
#   powershell -ExecutionPolicy Bypass -File training/pipeline_watchdog.ps1 -Register
# Remove when done:
#   Unregister-ScheduledTask -TaskName "SmartRouter-v21-pipeline" -Confirm:$false

param([switch]$Register, [switch]$Unregister)

$TaskName = "SmartRouter-v21-pipeline"
$Repo = "C:\Users\diffi\Desktop\SmartRouter-main"
$Result = "C:\Users\diffi\Desktop\SmartRouter\v21-재탐색-결과.md"
$WatchLog = Join-Path $Repo "build\watchdog.log"

if ($Register) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument ("-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"" +
                   (Join-Path $Repo "training\pipeline_watchdog.ps1") + "`"")
    # Every 20 minutes for 24 hours. Long enough to cover today, short enough
    # that it expires on its own instead of lingering on the machine.
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
        -RepetitionInterval (New-TimeSpan -Minutes 20) `
        -RepetitionDuration (New-TimeSpan -Hours 24)
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries:$false `
        -DontStopIfGoingOnBatteries:$false -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 24)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Description "v21 재탐색 파이프라인 감시자 (끝나면 자동 해제)" -Force | Out-Null
    "등록 완료: $TaskName (20분 간격, 24시간 후 만료)"
    exit 0
}

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    "해제 완료: $TaskName"
    exit 0
}

function Note($msg) {
    "[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $msg |
        Add-Content -Path $WatchLog -Encoding utf8
}

# Finished: take the task off the machine rather than leaving it to tick for a
# day against work that is already done.
if (Test-Path $Result) {
    Note "결과 파일 존재 -- 감시 종료, 태스크 해제"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    exit 0
}

# Already working. Checking for the search process rather than the wrapper
# shell is deliberate: the shell can be gone while a slice is still running.
$running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*search_standalone*' -or $_.CommandLine -like '*finalize_search*' -or
                   $_.CommandLine -like '*audit_operating_point*' }
if ($running) {
    Note ("실행 중 ({0}개) -- 아무것도 하지 않음" -f @($running).Count)
    exit 0
}

Note "실행 중이 아님 -- 파이프라인 재시작"
$bash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $bash)) { $bash = "bash.exe" }
Start-Process -FilePath $bash `
    -ArgumentList "-lc", "cd '$($Repo -replace '\\','/')' && bash training/run_v21_pipeline.sh" `
    -WindowStyle Hidden
Note "재시작 요청 보냄"
