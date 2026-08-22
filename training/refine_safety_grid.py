# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Fill in the safety values the coarse grid skipped.

The operating policy is: never exceed a 1.5% overrun bound, hold steady near
1.0%, tune aggressively below 0.5%. v16 sits at 0.3% on all three tiers, so
policy says there is room -- but the grid cannot reach it. SAFETY_GRID jumps
0.05 -> 0.10, and for fast those two points bound at 0.30% and 1.57%: one is
below the aggressive threshold, the next is already over the hard limit.
Everything the policy actually allows lies in the gap.

This re-bootstraps a fine grid between the chosen point and the first value
that broke the limit, holding each tier's risk multipliers at what the full
search chose, so only the cap moves.

    py -3.13 training/refine_safety_grid.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

from finalize_search import clopper_pearson_upper as cp  # noqa: E402
from search_standalone import TIER_ORDER, bootstrap, load_splits  # noqa: E402

# The bands the coarse grid skipped, per tier: from the value in use up to the
# first one measured over the 1.5% limit.
BANDS = {
    "fast": np.arange(0.08, 0.081, 0.01),
    "balanced": np.arange(0.80, 1.001, 0.04),
    "premium": np.arange(0.56, 0.561, 0.01),
}
LIMIT = 0.015
STEADY = 0.010


def main() -> int:
    chosen = json.loads((REPO_ROOT / "build/v16-tol0005.json").read_text(encoding="utf-8"))
    splits, data = load_splits(REPO_ROOT / "build/search-matrices.npz")
    budgets = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))

    for tier in TIER_ORDER:
        c = chosen[tier]
        risk = np.array([1.0, c["risk_mid"], c["risk_high"]])
        print(f"\n=== {tier}  risk=({c['risk_mid']}, {c['risk_high']})  "
              f"한도 {budgets[tier]}  [현재 safety={c['safety_ratio']}]")
        print(f"{'safety':>7}{'cap/light':>11}{'dev':>7}{'train':>7}{'95%상한':>10}{'점수':>9}  판정")
        for s in BANDS[tier]:
            rng = np.random.default_rng((31, TIER_ORDER.index(tier), int(round(s * 1000))))
            _, score, counts, k = bootstrap(
                splits, budget_multiplier=budgets[tier], safety_ratio=float(s),
                risk_multiplier=risk, high_cap_ratio=c.get("high_cap_ratio", 1.0),
                share_ratio=c.get("share_ratio", 1.0), rng=rng, k=3000, with_counts=True)
            bound = max(cp(int(x), k) for x in counts)
            verdict = ("금지" if bound > LIMIT else
                       "상한선(steady)" if bound > STEADY else
                       "여유" if bound > 0.005 else "공격 허용")
            mark = "  <== 현재" if abs(s - c["safety_ratio"]) < 1e-9 else ""
            print(f"{s:>7.2f}{1 + (budgets[tier] - 1) * s:>11.4f}"
                  f"{counts[0]:>7}{counts[1]:>7}{bound:>9.2%}{score:>9.4f}  {verdict}{mark}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
