#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0
#
# Everything that happens after six curve files land: calibrate, audit, bake,
# check, stress. One script because the order matters and because doing it by
# hand at 2am is how steps get skipped.
#
#   bash training/finish_arm.sh v22 training/tmp-v22-matrices.npz
#
# Leaves main and the other branches alone -- it bakes into the working tree,
# which is already on the arm's own branch.

set -u
cd "$(dirname "$0")/.." || exit 1

TAG="${1:?사용법: finish_arm.sh <태그> <행렬파일>}"
MATRICES="${2:?}"
PY=./.venv-data/Scripts/python
LOG="build/${TAG}-finish.log"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
: > "$LOG"

# The script bakes into the working tree, so the working tree must be on the
# arm's own branch. Being on a different one would bake this arm's coefficients
# into another arm's artifact -- silently, since both files are named the same.
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "SmartRouter-${TAG}" ]; then
  say "!! 브랜치 ${BRANCH} 인데 ${TAG} 를 마무리하려 함 -- 중단"
  say "   먼저: git checkout SmartRouter-${TAG}"
  exit 1
fi
say "브랜치 ${BRANCH} 확인"

curves=(build/${TAG}-curve-s*.json)
if [ "${#curves[@]}" -ne 6 ]; then
  say "!! 곡선 ${#curves[@]}/6 -- 중단"
  exit 1
fi
say "곡선 6개 확인"

# Several targets, because the curves cost the same whatever target is asked of
# them and a looser one is worth having on record.
for target in 0.003 0.005 0.010; do
  out="build/${TAG}-target-${target}.json"
  [ -f "$out" ] && continue
  $PY -u training/finalize_search.py --curves "${curves[@]}" --matrices "$MATRICES" \
      --overrun-target "$target" --out "$out" >> "$LOG" 2>&1 \
    && say "목표 ${target} 보정 완료" || say "!! 목표 ${target} 보정 실패"
done

# k=5000 against ONE fixed configuration: no selection, so the bound is a
# certificate rather than a corrected estimate.
if [ -f "build/${TAG}-target-0.003.json" ]; then
  say "k=5000 감사"
  $PY -u training/audit_operating_point.py --matrices "$MATRICES" \
      --config "build/${TAG}-target-0.003.json" -k 5000 \
      > "build/${TAG}-audit.log" 2>&1 && say "감사 완료" || say "!! 감사 실패"
fi

say "아티팩트에 파라미터 적용 + 굽기"
$PY training/apply_search.py --artifact training/artifact.json \
    --results "build/${TAG}-target-0.003.json" >> "$LOG" 2>&1 \
  && $PY training/bake_artifact.py --artifact training/artifact.json >> "$LOG" 2>&1 \
  && say "굽기 완료" || { say "!! 굽기 실패"; exit 1; }

say "공식 self-check (두 split)"
$PY training/verify_both_splits.py > "build/${TAG}-selfcheck.log" 2>&1 \
  && say "self-check 완료" || say "!! self-check 실패"

say "구성 스트레스"
$PY -u training/private_set_stress.py --params "build/${TAG}-target-0.003.json" \
    --matrices "$MATRICES" > "build/${TAG}-stress.log" 2>&1 \
  && say "스트레스 완료" || say "!! 스트레스 실패"

# The composition stress above varies WHICH episodes are in the batch. This
# one varies HOW EXPENSIVE they turn out to be -- and it is the test that
# decided v21d over v20b, because it is the one that finds a setting where
# all three tiers cross at once and the total score goes to zero. Running
# only the first let v22 look clean while it had three such settings.
say "비용 폭주 시나리오"
$PY -u training/stress_scenarios.py --params "build/${TAG}-target-0.003.json" --matrices "$MATRICES" > "build/${TAG}-scenarios.log" 2>&1 && say "시나리오 완료" || say "!! 시나리오 실패"

# The one number that decides adoption: does any setting take all three
# tiers at once? grep -c exits 1 when it finds nothing, so let it print its
# own 0 rather than appending one.
zero=$(grep -c "!.*!.*!" "build/${TAG}-scenarios.log" 2>/dev/null)
if [ "${zero:-0}" -gt 0 ]; then
  say "!! 총점 0 시나리오 ${zero}개 -- 채택 전에 반드시 확인"
else
  say "총점 0 시나리오 없음"
fi

say "완료. 로그: build/${TAG}-{audit,selfcheck,stress,scenarios}.log"
echo
echo "===== 감사 ====="; cat "build/${TAG}-audit.log" 2>/dev/null
echo; echo "===== self-check ====="; head -8 "build/${TAG}-selfcheck.log" 2>/dev/null
echo; echo "===== 스트레스 ====="; cat "build/${TAG}-stress.log" 2>/dev/null
echo; echo "===== 폭주 시나리오 ====="; cat "build/${TAG}-scenarios.log" 2>/dev/null
