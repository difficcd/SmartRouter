# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""How much does the CALIBRATION overfit the split it was chosen on?

Prediction overfitting was measured on a holdout: fit Train, predict Dev, and
the gains survived. This measures the other half, and it is the half we
actually tune -- the operating point. Pick the best point using Dev alone, then
read what it scores on Train, and vice versa. A small cross-split drop means
the choice transfers. A large one means we were fitting one split's quirks.

The both-splits row is what we ship, and it exists precisely to avoid the
failure the single-split rows expose. Quantifying that failure is the point:
"we calibrate on both splits" is a claim, and this turns it into a number.

Every point is evaluated with the REALIZED score (no resampling), because the
adoption criterion has always been the worst realized split rather than the
mean bootstrap the search optimises.

    py -3.13 training/calibration_overfit.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

from finalize_search import clopper_pearson_upper as cp  # noqa: E402
from search_standalone import TIER_ORDER, allocate, load_splits  # noqa: E402

WEIGHT = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
NAMES = ("dev", "train")
BOUND = 0.005


def main() -> int:
    splits, data = load_splits(REPO_ROOT / "training/tmp-v21-matrices.npz")
    budgets = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))
    rows = []
    for f in sorted(glob.glob(str(REPO_ROOT / "build/v21-curve-s*.json"))):
        rows += json.loads(Path(f).read_text(encoding="utf-8"))["rows"]

    cache: dict = {}

    def score_at(tier, rm, rh, s):
        key = (tier, rm, rh, s)
        if key in cache:
            return cache[key]
        risk = np.array([1.0, rm, rh])
        out = {}
        for name, (ps, pc, prc, rs, rc) in zip(NAMES, splits):
            ch = allocate(ps, pc, prc, budget_multiplier=budgets[tier],
                          safety_ratio=s, risk_multiplier=risk,
                          high_cap_ratio=1.0, share_ratio=1.0)
            total = rc[np.arange(len(rc)), ch].sum()
            light = rc[:, 0].sum()
            out[name] = (float(rs[np.arange(len(rs)), ch].mean())
                         if total <= light * budgets[tier] else 0.0)
        cache[key] = out
        return out

    points = {t: [] for t in TIER_ORDER}
    for r in rows:
        for s, _ov, _sc, counts, k in r["curve"]:
            points[r["tier"]].append((r["risk_mid"], r["risk_high"], s, counts, k))

    selectors = (("dev만", 0), ("train만", 1), ("양쪽(출하 방식)", None))
    totals = {label: {n: 0.0 for n in NAMES} for label, _ in selectors}
    print(f"{'선택 기준':>16}{'등급':>10}{'dev':>9}{'train':>9}{'교차 하락':>11}"
          f"   고른 지점", flush=True)
    print("-" * 78)
    for tier in TIER_ORDER:
        for label, sel in selectors:
            best = None
            for rm, rh, s, counts, k in points[tier]:
                # each selector applies the safety rule it would have applied
                # on its own evidence, so the comparison is about the choice
                # rather than about who was allowed to look at what
                if sel is None:
                    if max(cp(int(c), k) for c in counts) > BOUND:
                        continue
                    rank = lambda v: min(v.values())  # noqa: E731
                else:
                    if cp(int(counts[sel]), k) > BOUND:
                        continue
                    rank = lambda v: v[NAMES[sel]]  # noqa: E731
                value = score_at(tier, rm, rh, s)
                if best is None or rank(value) > rank(best[0]):
                    best = (value, rm, rh, s)
            value = best[0]
            for n in NAMES:
                totals[label][n] += WEIGHT[tier] * value[n]
            if sel == 0:
                drop = f"{value['train'] - value['dev']:+.4f}"
            elif sel == 1:
                drop = f"{value['dev'] - value['train']:+.4f}"
            else:
                drop = "-"
            print(f"{label:>16}{tier:>10}{value['dev']:>9.4f}{value['train']:>9.4f}"
                  f"{drop:>11}   risk=({best[1]},{best[2]}) s={best[3]}", flush=True)
        print(flush=True)

    print(f"{'선택 기준':>16}{'':>10}{'dev':>9}{'train':>9}{'최악':>11}")
    for label, _ in selectors:
        t = totals[label]
        print(f"{label:>16}{'':>10}{t['dev']:>9.4f}{t['train']:>9.4f}"
              f"{min(t.values()):>11.4f}")
    print("\n해석: 'dev만'/'train만'의 교차 하락이 크면 그 split에 과적합한 것이고,")
    print("      '양쪽'이 그보다 안정적이면 두-split 보정이 값을 하고 있다는 뜻이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
