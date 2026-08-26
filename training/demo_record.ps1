# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0
#
# 시연 영상 녹화용 스크립트. 화면 녹화를 켜고 이것만 실행하면 된다.
#
#   powershell -ExecutionPolicy Bypass -File training\demo_record.ps1
#
# GUI 가 없는 프로젝트라 보여줄 것은 세 가지다.
#   1) 이미지가 자격증명 없이 받아진다 (평가자가 처하는 조건)
#   2) 공식 자원 한도 -- 네트워크 차단, 읽기 전용 -- 에서 실제로 돈다
#   3) 보고서에 적은 점수가 그대로 재현된다

$ErrorActionPreference = "Continue"
$root   = Split-Path -Parent $PSScriptRoot
$digest = "ghcr.io/difficcd/smartrouter@sha256:b7e04a8cab897716da06a0daa63b017ae8d4b375f49cab206e9cbee3a6b642d8"

function Title($n, $text) {
  Write-Host ""
  Write-Host ("=" * 74) -ForegroundColor DarkGray
  Write-Host ("  [$n] $text") -ForegroundColor Cyan
  Write-Host ("=" * 74) -ForegroundColor DarkGray
  Write-Host ""
  Start-Sleep -Seconds 2
}
function Run($cmd) {
  Write-Host "> $cmd" -ForegroundColor Yellow
  Start-Sleep -Milliseconds 900
  Invoke-Expression $cmd
  Start-Sleep -Seconds 2
}

Clear-Host
Write-Host ""
Write-Host "  SmartRouter -- 프롬프트만 보고 세 LLM 중 하나를 고르는 라우터" -ForegroundColor White
Write-Host "  2026 오픈소스 개발자대회 / 지정과제(SK텔레콤) / 팀 difficcd" -ForegroundColor DarkGray
Write-Host ""
Start-Sleep -Seconds 3

# ------------------------------------------------------------------
Title 1 "이미지는 자격증명 없이 받을 수 있다 (평가자와 같은 조건)"
Run "docker logout ghcr.io"
Run "docker rmi $digest 2>`$null | Out-Null; docker pull --platform linux/arm64 $digest"

# ------------------------------------------------------------------
Title 2 "공식 자원 한도에서 실행 -- 네트워크 차단, 읽기 전용 루트"
Write-Host "  CPU 2개 / 메모리 2GB / --network none / --read-only / 등급당 90초" -ForegroundColor DarkGray
Write-Host ""
$out = Join-Path $root "build\demo-out"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Remove-Item "$out\*.json" -ErrorAction SilentlyContinue
$in = Join-Path $root "data\materialized\dev\inputs.json"

$sw = [Diagnostics.Stopwatch]::StartNew()
Run @"
docker run --rm --platform linux/arm64 ``
  --cpus 2 --memory 2g --network none --read-only --tmpfs /tmp:rw,size=64m ``
  -v "${in}:/challenge/input/inputs.json:ro" -v "${out}:/challenge/output" ``
  $digest ``
  --input /challenge/input/inputs.json --tier fast --output /challenge/output/fast.json
"@
$sw.Stop()
Write-Host ("  경과 {0:N1}초  (x86 위 QEMU 에뮬레이션. 네이티브 arm64 에서는 약 5초)" -f $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray
Start-Sleep -Seconds 2

Title 3 "네트워크가 정말로 차단되어 있는지 확인"
$probe = 'import socket; socket.create_connection((''1.1.1.1'',53), 2)'
Write-Host "> docker run --network none ... python3 -c `"소켓 연결 시도`"" -ForegroundColor Yellow
Start-Sleep -Milliseconds 900
# 2>&1 을 PowerShell 파이프로 흘리면 NativeCommandError 가 붉게 찍힌다.
# cmd 안에서 합쳐 받은 뒤 마지막 줄만 보여준다.
$msg = cmd /c "docker run --rm --platform linux/arm64 --network none --entrypoint python3 $digest -c ""$probe"" 2>&1"
$last = ($msg | Where-Object { $_ -match "Error|error" } | Select-Object -Last 1)
if (-not $last) { $last = ($msg | Select-Object -Last 1) }
Write-Host "  $last" -ForegroundColor Red
Write-Host ""
Write-Host "  -> 연결 자체가 불가능하다. 라우터는 외부 호출 없이 판단한다." -ForegroundColor DarkGray
Start-Sleep -Seconds 3

# ------------------------------------------------------------------
Title 4 "산출물 확인 -- 문항별로 모델 하나"
$f = Join-Path $out "fast.json"
Write-Host "> Get-Item build\demo-out/fast.json" -ForegroundColor Yellow
Start-Sleep -Milliseconds 700
$fi = Get-Item $f
Write-Host ("  {0}   {1:N0} 바이트" -f $fi.Name, $fi.Length)
Write-Host ""
Start-Sleep -Seconds 2
Run "& `"$root/.venv-data/Scripts/python.exe`" `"$root/training\demo_summary.py`" `"$f`""

# ------------------------------------------------------------------
Title 5 "보고서에 적은 점수가 그대로 재현된다"
Run "& `"$root/.venv-data/Scripts/python.exe`" `"$root\training\verify_both_splits.py`""

Write-Host ""
Write-Host ("=" * 74) -ForegroundColor DarkGray
Write-Host "  저장소  https://github.com/difficcd/SmartRouter" -ForegroundColor White
Write-Host "  이미지  ghcr.io/difficcd/smartrouter@sha256:b7e04a8c..." -ForegroundColor White
Write-Host ("=" * 74) -ForegroundColor DarkGray
Write-Host ""
