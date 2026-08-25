#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0
#
# Ask what merging a working branch into main would actually put in the public
# tree, before doing it.
#
#   bash training/preflight_merge.sh SmartRouter-v22
#
# The working branches carry things main must never have: the exported matrices
# (committed only so a Colab notebook can clone them), pipeline scripts holding
# absolute home-directory paths, and notebooks for slicings that were replaced.
# The repository's own policy test rejects all of these, so a straight merge
# turns CI red -- and puts internal paths in a public repository, which is the
# part that matters beyond CI.

set -u
cd "$(dirname "$0")/.." || exit 1
BRANCH="${1:?사용법: preflight_merge.sh <브랜치>}"

echo "main <- ${BRANCH} 병합 사전점검"
echo

bad=0

echo "[1] 공개 트리에 들어가면 안 되는 파일"
offenders=$(git diff --name-only main "$BRANCH" | grep -E '\.npz$|/tmp-|run_v2.*pipeline\.sh$' || true)
if [ -n "$offenders" ]; then
  while read -r f; do
    [ -z "$f" ] && continue
    sz=$(git cat-file -s "$(git rev-parse "$BRANCH:$f" 2>/dev/null)" 2>/dev/null || echo "?")
    echo "  !! $f  (${sz} 바이트)"
    bad=1
  done <<< "$offenders"
else
  echo "  없음"
fi

echo
echo "[2] 절대경로가 든 텍스트 파일"
found=0
while read -r f; do
  [ -z "$f" ] && continue
  case "$f" in *.npz|*.png|*.jpg) continue;; esac
  if git show "$BRANCH:$f" 2>/dev/null | grep -q "/Users/\|C:\\Users"; then
    echo "  !! $f"
    found=1; bad=1
  fi
done <<< "$(git diff --name-only main "$BRANCH")"
[ "$found" -eq 0 ] && echo "  없음"

echo
echo "[3] 대체되어 쓰이지 않는 노트북"
stale=$(git diff --name-only main "$BRANCH" | grep '^notebooks/' | grep -v "${BRANCH#SmartRouter-}" || true)
if [ -n "$stale" ]; then
  echo "$stale" | sed 's/^/  ?  /'
  echo "  (이 브랜치의 arm 것이 아닙니다. 남길지 판단하세요)"
else
  echo "  없음"
fi

echo
if [ "$bad" -ne 0 ]; then
  echo "판정: 그대로 병합하면 안 됩니다."
  echo "  파일을 골라 가져오세요:  git checkout ${BRANCH} -- <경로들>"
  echo "  그 다음 정책 테스트로 확인:"
  echo "    PYTHONPATH=src python -m pytest tests/test_repository_policy.py -q"
  exit 1
fi
echo "판정: 병합해도 공개 트리 정책에 걸리는 것이 없습니다."
