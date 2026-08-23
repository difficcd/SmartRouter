# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Add a stress-survival requirement to the calibration choice.

The bootstrap picks a safety point by replaying observed variation, which
means it never asks what happens if the batch-level cost multiple itself
moves. stress_scenarios.py showed that question has teeth: on main, balanced fails
once ax31's batch multiple reaches 3.0 -- a 39% shift, far outside the 2.5%
the two splits show, but also exactly the bound our share-based constraint
assumes.

This walks the same curve the bootstrap uses and reports, for every safety
value, BOTH numbers: the Clopper-Pearson overrun bound and the batch multiple
at which the tier stops passing. Choosing on both is how a point gets to be
safe against variation we measured and against variation we did not.

    py -3.13 training/stress_filter.py --curves build/colab/curve-s*.json \
        --matrices build/search-matrices.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

from finalize_search import clopper_pearson_upper  # noqa: E402
from search_standalone import TIER_ORDER, allocate, load_splits  # noqa: E402

STRESS_MULTIPLES = (2.5, 3.0, 4.0, 6.0)


def break_point(tier, budget, cfg, splits):
    """Lowest stressed ax31 batch multiple at which either split overruns."""
    for m31 in STRESS_MULTIPLES:
        for ps, pc, prc, rs, rc in splits:
            base31 = rc[:, 1].sum() / rc[:, 0].sum()
            basek1 = rc[:, 2].sum() / rc[:, 0].sum()
            damaged = rc.copy()
            damaged[:, 1] *= m31 / base31
            damaged[:, 2] *= (m31 / base31 * basek1) / basek1
            choice = allocate(ps, pc, prc, budget_multiplier=budget, **cfg)
            n = np.arange(len(choice))
            if damaged[n, choice].sum() / damaged[:, 0].sum() > budget:
                return m31
    return None  # survived everything tried


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--curves", type=Path, nargs="+", required=True)
    parser.add_argument("--matrices", required=True)
    parser.add_argument("--chosen", type=Path, required=True,
                        help="현재 선택된 파라미터 JSON (risk 배수를 여기서 가져온다)")
    args = parser.parse_args(argv)

    rows, budgets = [], {}
    for path in args.curves:
        chunk = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(chunk["rows"])
        budgets.update(chunk["budget_multipliers"])
    chosen = json.loads(args.chosen.read_text(encoding="utf-8"))
    splits, _ = load_splits(args.matrices)

    for tier in TIER_ORDER:
        c = chosen[tier]
        curve = next(r["curve"] for r in rows
                     if r["tier"] == tier and r["risk_mid"] == c["risk_mid"]
                     and r["risk_high"] == c["risk_high"])
        print(f"\n=== {tier}  risk=({c['risk_mid']}, {c['risk_high']})  "
              f"한도 {budgets[tier]}  [현재 선택: safety={c['safety_ratio']}]")
        print(f"{'safety':>7}{'95%상한':>10}{'점수':>9}{'붕괴 배수':>11}")
        for row in curve:
            safety, _, score, counts, k = row
            bound = max(clopper_pearson_upper(int(x), int(k)) for x in counts)
            if bound > 0.003:
                continue
            cfg = dict(safety_ratio=safety,
                       risk_multiplier=np.array([1.0, c["risk_mid"], c["risk_high"]]),
                       high_cap_ratio=c.get("high_cap_ratio", 1.0),
                       share_ratio=c.get("share_ratio", 1.0))
            bp = break_point(tier, budgets[tier], cfg, splits)
            mark = "  <== 현재" if abs(safety - c["safety_ratio"]) < 1e-9 else ""
            print(f"{safety:>7.2f}{bound:>9.3%}{score:>9.4f}"
                  f"{('>' + str(STRESS_MULTIPLES[-1])) if bp is None else f'{bp:.1f}':>11}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
