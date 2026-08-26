#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0
#
# 네이티브 리눅스에서 컨테이너 실행 시간을 잰다.
#
# Windows(Docker Desktop + QEMU)에서 잰 값에는 에뮬레이션·VM·9p 파일시스템이
# 겹쳐 있어 90초 한도 판정에 쓸 수 없다. 실측된 에뮬레이션 배율은 10.5배다
# (같은 이미지·입력으로 네이티브 amd64 14.9초 vs QEMU arm64 155.8초).
# 이 스크립트는 같은 조건을 네이티브 커널에서 재현해 그 층들을 벗겨낸다.
#
#   bash training/native_timing.sh              # Dev 880문항
#   bash training/native_timing.sh --combined   # Train+Dev 2,640문항
#
# 자세한 배경과 문제 해결은 training/NATIVE_TIMING_HANDOFF.md 참고.
set -u
cd "$(dirname "$0")/.." || exit 1

COMBINED=0
[ "${1:-}" = "--combined" ] && COMBINED=1

ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|arm64) PLAT=linux/arm64; NOTE="공식 환경과 같은 아키텍처 -- 실측값" ;;
  x86_64|amd64)  PLAT=linux/amd64; NOTE="공식 환경(arm64)과 다름 -- 대리값" ;;
  *) echo "모르는 아키텍처: $ARCH"; exit 1 ;;
esac
echo "장비 $ARCH  ->  $PLAT"
echo "$NOTE"
command -v lscpu >/dev/null && lscpu | grep -m1 'Model name' | sed 's/^/CPU  /'
echo

DEV=data/materialized/dev/inputs.json
TRAIN=data/materialized/train/inputs.json
[ -f "$DEV" ] || { echo "!! $DEV 없음 -- HANDOFF 문서의 '준비물 3' 참고"; exit 1; }

if [ "$COMBINED" = "1" ]; then
  [ -f "$TRAIN" ] || { echo "!! $TRAIN 없음 -- --combined 에는 Train 입력도 필요"; exit 1; }
  IN=build/native-combined.json
  mkdir -p build
  python3 - "$TRAIN" "$DEV" "$IN" <<'PY'
import json, sys
tr, dv, out = sys.argv[1], sys.argv[2], sys.argv[3]
a = json.load(open(tr, encoding="utf-8")); b = json.load(open(dv, encoding="utf-8"))
a["episodes"] = a["episodes"] + b["episodes"]; a["split"] = "train+dev"
json.dump(a, open(out, "w", encoding="utf-8"), ensure_ascii=False)
PY
  [ -f "$IN" ] || { echo "!! 결합 입력 생성 실패"; exit 1; }
else
  IN="$DEV"
fi

N=$(python3 -c "import json;print(len(json.load(open('$IN',encoding='utf-8'))['episodes']))")
echo "입력 $IN  ($N 문항)"

echo "이미지 빌드..."
docker buildx build --platform "$PLAT" --load --provenance=false --sbom=false \
  --file container/Dockerfile --tag smartrouter:native . >/dev/null 2>&1 \
  || { echo "!! 빌드 실패 -- docker buildx ls 로 $PLAT 지원을 확인하세요"; exit 1; }
echo "빌드 완료: $(docker image inspect smartrouter:native --format '{{.Architecture}}  {{.Size}}바이트')"
echo

OUT="$(mktemp -d)"
# 컨테이너는 UID 65532 로 돈다. mktemp -d 는 0700 이라 그대로 두면
# 출력 파일을 못 쓰고 종료코드 2 로 끝난다 -- 연산은 다 끝낸 뒤라
# 시간만 보면 정상으로 보여서 놓치기 쉽다.
chmod 777 "$OUT"
printf '%-10s %9s %8s  %s\n' 등급 초 한도 판정
FAILED=0
for tier in fast balanced premium; do
  s=$(date +%s%N)
  # 공식 자원 한도. SELinux 장비(Rocky/Fedora)에서 마운트가 거부되면
  # 아래 두 -v 인자의 ro / 끝에 ,z 를 덧붙이세요.
  docker run --rm --platform "$PLAT" \
    --cpus 2 --memory 2g --memory-swap 2g --network none --read-only \
    --tmpfs /tmp:rw,size=64m \
    -v "$(pwd)/$IN:/challenge/input/inputs.json:ro" \
    -v "$OUT:/challenge/output" \
    smartrouter:native \
    --input /challenge/input/inputs.json --tier "$tier" \
    --output "/challenge/output/$tier.json" >/dev/null 2>&1
  rc=$?
  e=$(date +%s%N)
  t=$(awk "BEGIN{printf \"%.1f\", ($e - $s)/1000000000}")
  if [ $rc -ne 0 ]; then v="실패(종료 $rc)"; FAILED=1
  elif awk "BEGIN{exit !($t > 90)}"; then v="초과"; FAILED=1
  else v="통과"; fi
  printf '%-10s %9s %8s  %s\n' "$tier" "$t" 90 "$v"
done
echo
echo "출력 디렉터리: $OUT"
ls -la "$OUT" 2>/dev/null | tail -n +2
echo
echo "주의: 시간에는 컨테이너 기동(약 0.3~1초)이 포함됩니다."
[ "$FAILED" = "1" ] && echo "주의: 실패/초과가 있습니다. 고치지 말고 그대로 보고하세요."
exit 0
