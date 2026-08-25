#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0
#
# Re-search for v22, locally and unattended.
#
# Same shape as the v21 runner: six slices so a crash costs one slice rather
# than the whole grid, and a rerun skips any slice whose curve file already
# parses with its expected 24 rows. Eight workers rather than three because the
# package temperature is readable now (LibreHardwareMonitor's web server) and
# the cooldown waits on a measurement instead of a stopwatch.
#
#   bash training/run_v22_pipeline.sh        # start, or resume after a crash
#
# Progress: build/v22-pipeline.log
# Result:   the 결과 file named below

set -u
cd "$(dirname "$0")/.." || exit 1

PY=./.venv-data/Scripts/python
MATRICES=training/tmp-v22-matrices.npz
TAG=v22
WORKERS=8
RESUME_TEMP=72     # resume once the package is back under this
MAX_COOL=600
LOG=build/v22-pipeline.log
RESULT="C:/Users/diffi/Desktop/SmartRouter/v22-재탐색-결과.md"

SLICES=(
  "s1 fast     1.0 1.5 2.0 2.5"
  "s2 fast     3.0 4.0 5.0 6.0"
  "s3 balanced 1.0 1.5 2.0 2.5"
  "s4 balanced 3.0 4.0 5.0 6.0"
  "s5 premium  1.0 1.5 2.0 2.5"
  "s6 premium  3.0 4.0 5.0 6.0"
)
EXPECT_ROWS=24

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
cpu_temp() { powershell -NoProfile -ExecutionPolicy Bypass -File training/cpu_temp.ps1 2>/dev/null; }

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

cool_down() {
  local t waited=0
  t="$(cpu_temp)"
  if [ -z "$t" ]; then
    say "냉각 120초 (온도 못 읽음)"
    sleep 120
    return
  fi
  if [ "${t%.*}" -lt "$RESUME_TEMP" ]; then
    say "냉각 불필요 -- ${t}°C"
    return
  fi
  say "냉각 -- 현재 ${t}°C, ${RESUME_TEMP}°C 아래까지"
  while [ "$waited" -lt "$MAX_COOL" ]; do
    sleep 20
    waited=$((waited + 20))
    t="$(cpu_temp)"
    [ -z "$t" ] && break
    [ "${t%.*}" -lt "$RESUME_TEMP" ] && { say "냉각 완료 -- ${t}°C"; return; }
  done
  say "냉각 종료 -- ${t}°C (상한 도달)"
}

say "v22 재탐색 시작 -- 워커 ${WORKERS}, 6조각, 시작 온도 $(cpu_temp)°C"

for row in "${SLICES[@]}"; do
  set -- $row
  s="$1"; tier="$2"; shift 2
  curve="build/${TAG}-curve-${s}.json"
  if slice_done "$curve"; then
    say "${s} (${tier}): 이미 완료 -- 건너뜀"
    continue
  fi
  say "${s} (${tier}) risk_high=$*: 시작 ($(cpu_temp)°C)"
  $PY -u training/search_standalone.py \
      --matrices "$MATRICES" --tiers "$tier" --risk-high "$@" \
      --workers "$WORKERS" \
      --out "build/${TAG}-search-${s}.json" \
      --curve-out "$curve" >> "$LOG" 2>&1
  if slice_done "$curve"; then
    say "${s}: 완료 ($(cpu_temp)°C)"
  else
    say "!! ${s}: 실패 -- 재실행하면 이 조각부터"
    rm -f "$curve"
  fi
  cool_down
done

curves=()
for row in "${SLICES[@]}"; do
  set -- $row
  [ -f "build/${TAG}-curve-${1}.json" ] && curves+=("build/${TAG}-curve-${1}.json")
done
if [ "${#curves[@]}" -ne 6 ]; then
  say "!! 조각 ${#curves[@]}/6 -- 보정 생략"
  exit 1
fi

for target in 0.003 0.005 0.010; do
  [ -f "build/${TAG}-target-${target}.json" ] && continue
  $PY -u training/finalize_search.py --curves "${curves[@]}" --matrices "$MATRICES" \
      --overrun-target "$target" --out "build/${TAG}-target-${target}.json" >> "$LOG" 2>&1 \
    && say "목표 ${target} 보정 완료" || say "!! 목표 ${target} 보정 실패"
done

if [ -f "build/${TAG}-target-0.003.json" ]; then
  say "k=5000 감사 시작"
  $PY -u training/audit_operating_point.py --matrices "$MATRICES" \
      --config "build/${TAG}-target-0.003.json" -k 5000 \
      > "build/${TAG}-audit.log" 2>&1 && say "감사 완료" || say "!! 감사 실패"
fi

{
  echo "# v22 재탐색 결과"
  echo
  echo "작성: $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "해시 블록 L1 정규화(==님 아이디어)를 넣은 뒤의 첫 재보정입니다."
  echo "\`main\`(v20b)과 \`SmartRouter-v21\`(v21d)은 손대지 않았습니다."
  echo
  echo "## 기준점"
  echo
  echo "| 구성 | 실측 최악 split | k=5000 최악 상한 | 구성 실패(36칸) |"
  echo "|---|---|---|---|"
  echo "| main (v20b) | 0.662898 | 0.364% | 5건 |"
  echo "| v21d | 0.660384 | 0.210% | 0건 |"
  echo
  echo "## v22 보정 결과"
  echo
  echo "| 초과확률 목표 | 가중합 | fast | balanced | premium |"
  echo "|---|---|---|---|---|"
  for target in 0.003 0.005 0.010; do
    f="build/${TAG}-target-${target}.json"
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
  if [ -f "build/${TAG}-audit.log" ]; then
    echo "감사 (k=5000, 선택 없는 고정 구성):"
    echo
    echo '```'
    cat "build/${TAG}-audit.log"
    echo '```'
    echo
  fi
  echo "## 다음 단계"
  echo
  echo "\`apply_search\` → \`bake_artifact\` → \`verify_both_splits\` → 구성 스윕 36칸."
  echo "v21d를 이기지 못하면 v21d를 제출합니다."
} > "$RESULT" 2>&1

say "완료. 결과: $RESULT"
