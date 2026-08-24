#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0
#
# Unattended re-search for v21: both arms, end to end, resumable.
#
# The guarantee this script offers is NOT "it will not die". nohup only blocks
# SIGHUP, and a background job did get killed when a session ended yesterday.
# What it offers instead is "dying costs one slice": the grid is cut into
# twelve chunks, each writes its own curve file, and a rerun skips every chunk
# already on disk. Worst case a crash loses ~40 minutes, not four hours.
#
# It deliberately does NOT bake. Baking edits heuristic.py, and which arm
# deserves that is a judgement made after comparing both audits. main stays
# untouched, so a failure here costs time and nothing else.
#
#   bash training/run_v21_pipeline.sh          # start, or resume after a crash
#
# Progress: build/pipeline.log, and the 진행 file named below
# Result:   the 결과 file named below

set -u
cd "$(dirname "$0")/.." || exit 1

PY=./.venv-data/Scripts/python
# 3 of 16 logical cores. Temperature is not readable without admin rights
# (MSAcpi_ThermalZoneTemperature returns access denied), so this is deliberately
# far below what the box could take rather than tuned against a measurement --
# the machine is a laptop, on mains power today, and reported 93C while an
# earlier 8-worker run was still alive alongside a 4-worker one. Twelve
# workers caused that; three plus a four-minute idle between chunks is the
# conservative setting chosen in its place. There are 17 hours available and
# roughly 14 hours of work at this rate, so throughput is not the binding
# constraint -- heat is.
WORKERS=3
COOLDOWN=240       # fallback idle between chunks when no reading is available
RESUME_TEMP=74     # resume once the package is back under this
MAX_COOL=900       # but never wait longer than this for it

NOTES_DIR="C:/Users/diffi/Desktop/SmartRouter"
RESULT="$NOTES_DIR/v21-재탐색-결과.md"
PROGRESS="$NOTES_DIR/v21-진행상황.md"
LOG=build/pipeline.log

SLICES=(
  "s1 fast     1.0 1.5 2.0 2.5"
  "s2 fast     3.0 4.0 5.0 6.0"
  "s3 balanced 1.0 1.5 2.0 2.5"
  "s4 balanced 3.0 4.0 5.0 6.0"
  "s5 premium  1.0 1.5 2.0 2.5"
  "s6 premium  3.0 4.0 5.0 6.0"
)
EXPECT_ROWS=24     # 4 risk_high x 6 risk_mid

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# A curve file counts as done only if it parses AND holds the expected number
# of combinations. A half-written file from a crash would otherwise be skipped
# on resume and silently shrink the grid.
slice_done() {
  [ -f "$1" ] || return 1
  $PY -c "
import json, sys
try:
    rows = json.load(open(sys.argv[1], encoding='utf-8'))['rows']
except Exception:
    sys.exit(1)
sys.exit(0 if len(rows) == int(sys.argv[2]) else 1)
" "$1" "$EXPECT_ROWS" 2>/dev/null
}

write_progress() {
  {
    echo "# v21 재탐색 진행 상황"
    echo
    echo "갱신: $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "| 팔 | 조각 | 등급 | 상태 |"
    echo "|---|---|---|---|"
    for tag in v21 v21bag v21nb; do
      for row in "${SLICES[@]}"; do
        set -- $row
        local s="$1" tier="$2"
        if slice_done "build/${tag}-curve-${s}.json"; then
          echo "| $tag | $s | $tier | 완료 |"
        else
          echo "| $tag | $s | $tier | 대기/진행 중 |"
        fi
      done
    done
    echo
    echo "완료되면 \`v21-재탐색-결과.md\`가 생깁니다."
  } > "$PROGRESS" 2>/dev/null
}

run_slice() {
  local tag="$1" matrices="$2" s="$3" tier="$4"; shift 4
  local curve="build/${tag}-curve-${s}.json"
  if slice_done "$curve"; then
    say "${tag}/${s} (${tier}): 이미 완료 -- 건너뜀"
    return 0
  fi
  say "${tag}/${s} (${tier}) risk_high=$*: 시작 (현재 $(cpu_temp)°C)"
  $PY -u training/search_standalone.py \
      --matrices "$matrices" --tiers "$tier" --risk-high "$@" \
      --workers "$WORKERS" \
      --out "build/${tag}-search-${s}.json" \
      --curve-out "$curve" >> "$LOG" 2>&1
  if slice_done "$curve"; then
    say "${tag}/${s}: 완료"
  else
    say "!! ${tag}/${s}: 실패 (재실행하면 이 조각부터 다시 시도)"
    rm -f "$curve"
  fi
  write_progress
  cool_down
}

# Wait on a measurement rather than a stopwatch when one is available.
# LibreHardwareMonitor's web server gives the package temperature; when it is
# not running, cpu_temp.ps1 prints nothing and this falls back to the fixed
# idle. It never invents a number -- an earlier session called the machine
# "safe" from a CPU-utilisation proxy while it sat at 95C.
cpu_temp() {
  powershell -NoProfile -ExecutionPolicy Bypass -File training/cpu_temp.ps1 2>/dev/null
}

cool_down() {
  local t
  t="$(cpu_temp)"
  if [ -z "$t" ]; then
    say "냉각 ${COOLDOWN}초 (온도 못 읽음 -- 고정 대기)"
    sleep "$COOLDOWN"
    return
  fi
  say "냉각 시작 -- 현재 ${t}°C, ${RESUME_TEMP}°C 아래로 내려가면 재개 (최대 ${MAX_COOL}초)"
  local waited=0
  while [ "$waited" -lt "$MAX_COOL" ]; do
    sleep 30
    waited=$((waited + 30))
    t="$(cpu_temp)"
    [ -z "$t" ] && break
    if [ "${t%.*}" -lt "$RESUME_TEMP" ]; then
      say "냉각 완료 -- ${t}°C (${waited}초 대기)"
      return
    fi
  done
  say "냉각 종료 -- ${t}°C (${waited}초 대기, 상한 도달)"
}

finish_arm() {
  local tag="$1" matrices="$2"
  local curves=()
  for row in "${SLICES[@]}"; do
    set -- $row
    [ -f "build/${tag}-curve-${1}.json" ] && curves+=("build/${tag}-curve-${1}.json")
  done
  if [ "${#curves[@]}" -ne 6 ]; then
    say "!! ${tag}: 조각 ${#curves[@]}/6 -- 보정 생략"
    return 1
  fi
  # Curves cost the same whatever target is asked of them, so record several.
  # 0.003 is what v20b shipped at; the looser two re-answer "is our
  # conservatism costing score" against the NEW predictor, because the old
  # answer (it is not) was measured on the broken one.
  for target in 0.003 0.005 0.010; do
    [ -f "build/${tag}-target-${target}.json" ] && continue
    $PY -u training/finalize_search.py \
        --curves "${curves[@]}" --matrices "$matrices" \
        --overrun-target "$target" \
        --out "build/${tag}-target-${target}.json" >> "$LOG" 2>&1 \
      && say "${tag}: 목표 ${target} 보정 완료" \
      || say "!! ${tag}: 목표 ${target} 보정 실패"
  done
  # k=5000 against ONE fixed configuration, so the bound is a certificate
  # rather than a selection-corrected estimate.
  if [ -f "build/${tag}-target-0.003.json" ] && [ ! -f "build/${tag}-audit.log" ]; then
    say "${tag}: k=5000 감사 시작"
    $PY -u training/audit_operating_point.py \
        --matrices "$matrices" --config "build/${tag}-target-0.003.json" \
        -k 5000 > "build/${tag}-audit.log" 2>&1 \
      && say "${tag}: 감사 완료" || say "!! ${tag}: 감사 실패"
  fi
}

say "시작/재개 -- 워커 ${WORKERS}, 냉각 ${COOLDOWN}초, 12조각"
write_progress

for row in "${SLICES[@]}"; do
  set -- $row
  run_slice v21 training/tmp-v21-matrices.npz "$@"
done
finish_arm v21 training/tmp-v21-matrices.npz

for row in "${SLICES[@]}"; do
  set -- $row
  run_slice v21bag training/tmp-v21bag-matrices.npz "$@"
done
finish_arm v21bag training/tmp-v21bag-matrices.npz

# The arm that was never built: a properly trained 266-feature model. v20b does
# not ship the trigonometric basis, it ships a 326-coefficient fit evaluated on
# 266 features, so "v20b vs v21" compared a misspecified linear-ish model
# against a correct nonlinear one and left this cell empty. It matters because
# v21 loses badly on composition shift -- premium 3.2498 against v20b's 2.5478
# on a length-skewed batch -- and a nonlinear transform of length-derived
# features is exactly what should degrade there.
for row in "${SLICES[@]}"; do
  set -- $row
  run_slice v21nb training/tmp-v21nb-matrices.npz "$@"
done
finish_arm v21nb training/tmp-v21nb-matrices.npz

say "요약 작성"
{
  echo "# v21 재탐색 결과"
  echo
  echo "작성: $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "삼각함수 기저가 추론에서 호출된 적이 없던 버그를 고친 뒤의 첫 재보정입니다."
  echo "\`main\`(v20b)은 손대지 않았습니다 — 굽기는 두 팔을 비교한 뒤의 판단이라"
  echo "사람이 합니다."
  echo
  echo "## 기준점"
  echo
  echo "| 구성 | 가중합 |"
  echo "|---|---|"
  echo "| main (v20b, 버그 있는 상태) | 0.666941 |"
  echo
  for tag in v21 v21bag v21nb; do
    label="v21 (버그 수정, 삼각함수 326)"
    [ "$tag" = "v21bag" ] && label="v21 + 배깅 300"
    [ "$tag" = "v21nb" ] && label="266 특징 (삼각함수 없음, 정상 학습)"
    echo "## $label"
    echo
    echo "| 초과확률 목표 | 가중합 | fast | balanced | premium |"
    echo "|---|---|---|---|---|"
    for target in 0.003 0.005 0.010; do
      f="build/${tag}-target-${target}.json"
      if [ -f "$f" ]; then
        $PY - "$f" "$target" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
w = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
total = sum(w[k] * d[k]["score"] for k in w)
cells = " | ".join(f"{d[k]['score']:.4f} (s={d[k]['safety_ratio']})"
                   for k in ("fast", "balanced", "premium"))
print(f"| {sys.argv[2]} | **{total:.6f}** | {cells} |")
PYEOF
      else
        echo "| $target | (없음) | | | |"
      fi
    done
    echo
    if [ -f "build/${tag}-audit.log" ]; then
      echo "감사 (k=5000, 선택 없는 고정 구성 -- 보정이 아니라 증명서):"
      echo
      echo '```'
      cat "build/${tag}-audit.log"
      echo '```'
      echo
    fi
  done
  echo "## 판정 기준 (측정 전에 박아둔 것)"
  echo
  echo "| 비교 | 의미 |"
  echo "|---|---|"
  echo "| v21 − 0.666941 | 삼각함수 수정의 **실제** 이득. 지금까지 우리가 받은 적 없는 값 |"
  echo "| v21bag − v21 ≥ +0.003 | 배깅 채택 |"
  echo "| +0.001 ~ +0.003 | 채택 (목적2 근거 보강) |"
  echo "| −0.001 ~ +0.001 (중립) | 최악 상한이 안 나빠지면 채택 |"
  echo "| < −0.001 | 배깅 기각 |"
  echo
  echo "## 다음 단계 (사람 판단 필요)"
  echo
  echo "이긴 팔로: \`apply_search\` → \`bake_artifact\` → \`verify_both_splits\`"
  echo "→ \`stress_scenarios\` → \`private_set_stress\`. 약 20분."
} > "$RESULT" 2>&1

write_progress
say "완료. 결과: $RESULT"
