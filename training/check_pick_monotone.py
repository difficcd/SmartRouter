# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Flag an operating point that a bootstrap draw made look safe by accident.

Spending less of the budget cannot make a limit easier to cross. So along one
risk pair's safety curve the overrun counts must not fall as safety rises --
any dip is bootstrap noise, and a selector that maximises score will walk
straight into it, because a dip is exactly where a high score meets a low
measured overrun.

v23's premium was picked at safety 0.30, sitting between 0.25 with one overrun
and 0.40 with twenty-three. A k=5000 audit put its 95% upper bound at 0.721%,
against a 0.3% target. v22's premium at the same nominal safety sat at the end
of a genuine run of zeros -- 0.20, 0.25, 0.30 all clean -- and audited at
0.095%. The curve shape said which was which before either audit ran.

    py -3.13 training/check_pick_monotone.py --params build/v23-target-0.003.json --tag v23
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TIERS = ("fast", "balanced", "premium")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", required=True, help="finalize_search 결과 JSON")
    parser.add_argument("--tag", required=True, help="곡선 파일 접두사 (v22, v23 ...)")
    parser.add_argument("--curve-dir", type=Path, default=REPO_ROOT / "build")
    args = parser.parse_args(argv)

    chosen = json.loads(Path(args.params).read_text(encoding="utf-8"))
    rows = []
    for path in sorted(glob.glob(str(args.curve_dir / f"{args.tag}-curve-s*.json"))):
        rows += json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
    if not rows:
        print(f"{args.tag} 곡선을 찾지 못했습니다.")
        return 1

    suspect = 0
    for tier in TIERS:
        cfg = chosen[tier]
        match = [r for r in rows if r["tier"] == tier
                 and abs(r["risk_mid"] - cfg["risk_mid"]) < 1e-9
                 and abs(r["risk_high"] - cfg["risk_high"]) < 1e-9]
        if len(match) != 1:
            print(f"!! {tier}: 곡선 {len(match)}개 -- 건너뜁니다")
            continue
        curve = sorted(match[0]["curve"], key=lambda row: row[0])
        pick = cfg["safety_ratio"]
        here = next(row for row in curve if abs(row[0] - pick) < 1e-9)
        here_worst = max(here[3])

        # Any lower-safety point with MORE overruns than the pick is the tell.
        below = [row for row in curve if row[0] < pick and max(row[3]) > here_worst]
        above = [row for row in curve if row[0] > pick][:1]
        nxt = max(above[0][3]) if above else None

        if below:
            suspect += 1
            worst = max(below, key=lambda row: max(row[3]))
            print(f"!! {tier}: safety {pick:.2f} 에서 초과 {here[3]}, "
                  f"그런데 더 낮은 safety {worst[0]:.2f} 에서는 {worst[3]}")
            print(f"   예산을 덜 쓰는데 더 자주 넘길 수는 없습니다. 이 0은 잡음입니다.")
            if nxt is not None:
                print(f"   (바로 위 지점은 {nxt}회) -- k=5000 감사를 반드시 확인하세요")
        else:
            run = [row for row in curve if row[0] <= pick and max(row[3]) <= here_worst]
            print(f"OK {tier}: safety {pick:.2f}, 아래로 {len(run)}개 지점이 "
                  f"모두 {here_worst} 이하 -- 고원의 끝입니다")

    print()
    if suspect:
        print(f"의심 {suspect}개 등급. 잡음으로 뽑힌 지점일 수 있습니다.")
        return 1
    print("모든 등급이 단조로운 고원 위에 있습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
