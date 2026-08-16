<!--
SPDX-FileCopyrightText: Copyright 2026 difficcd
SPDX-License-Identifier: Apache-2.0
-->

# 이번 커밋(7707838)에서 바뀐 코드만 모아둔 폴더

직접 주석 달면서 읽으려고 뽑아놓은 스크래치 사본입니다. 실제 저장소 파일이 아니라
**읽기/필기 전용 복사본**이니 여기 고쳐도 진짜 라우터 동작에는 영향 없습니다.

원본 커밋: `git show 7707838` / `git diff 3cccbf6 7707838`

| 파일 | 원본 경로 | 성격 |
|---|---|---|
| `01-gitignore-added-lines.txt` | `.gitignore` | 기존 파일에 몇 줄 추가 |
| `02-entrypoint.py` | `container/entrypoint.py` | 기존 파일, import 대상 한 줄 교체 (전체 13줄이라 통째로) |
| `03-heuristic-additions.py` | `src/ossp_router/heuristic.py` (227~2750행) | 기존 파일 맨 아래 새로 이어붙인 섹션. **학습된 JSON 블록(366~2571행, 2,205줄)은 순수 숫자 데이터라 생략**하고 자리만 표시해둠 — 구조는 `training/artifact.json` 참고 |
| `04-train_router.py` | `training/train_router.py` | 신규 파일, 전체 그대로 |
| `05-calibrate_safety.py` | `training/calibrate_safety.py` | 신규 파일, 전체 그대로 |
| `06-bake_artifact.py` | `training/bake_artifact.py` | 신규 파일, 전체 그대로 |

읽는 순서 추천: 02 → 04 → 05 → 06 → 03 (지금까지 대화에서 짚은 순서와 동일 —
실행 진입점 다음, 학습 파이프라인 세 단계, 마지막이 실제 서빙 로직).

각 파일에 원본 줄번호를 주석으로 남겨뒀으니, 막히면 실제 파일에서
`Ctrl+G`(줄 이동)로 바로 대조해볼 수 있습니다.
