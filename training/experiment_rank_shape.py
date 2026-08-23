# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Reshape the cost the ranking sees, leaving the budget arithmetic alone.

==님's idea: penalize expensive episodes more than proportionally, so the
router prefers several cheap promotions over one costly one. That directly
targets the failure this project keeps running into -- a single episode
carrying 66% of a tier's excess.

The reason it can be done without distorting anything: the allocator already
keeps two cost estimates apart. `pred_rank_costs` decides which episodes are
worth promoting; `pred_costs` decides whether the batch fits. Reshaping only
the first changes preference order while the budget check stays exactly linear,
which is what the real constraint is.

Four shapes, because they say different things about where the penalty should
bite:

  power    c**a          -- everything expensive pays more, smoothly
  blend    (1-w)c + w*c**a  -- ==님's "fully convex vs correcting with convex";
                              the convex term is a correction, not a replacement
  relu     c + b*max(0, c - t)  -- normal episodes untouched, only the tail pays
  sigmoid  c * (1 + b*S((c-t)/s)) -- like relu but with a soft knee

Reported against what actually matters: score, the Clopper-Pearson overrun
bound, and how concentrated the excess is.

    py -3.13 training/experiment_rank_shape.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

from finalize_search import clopper_pearson_upper as cp  # noqa: E402
from search_standalone import SAFETY_GRID, TIER_ORDER, allocate, bootstrap, load_splits  # noqa: E402


def shapes():
    """(label, transform) pairs. Transforms act on a normalized cost matrix."""
    out = [("선형 (현재)", lambda c: c)]
    for a in (1.15, 1.3, 1.5):
        out.append((f"power a={a}", lambda c, a=a: c ** a))
    for w in (0.3, 0.6):
        out.append((f"blend w={w} a=1.3", lambda c, w=w: (1 - w) * c + w * c ** 1.3))
    for t, b in ((2.0, 1.0), (4.0, 2.0)):
        out.append((f"relu t={t} b={b}",
                    lambda c, t=t, b=b: c + b * np.maximum(0.0, c - t)))
    out.append(("sigmoid t=3 b=1.5",
                lambda c: c * (1.0 + 1.5 / (1.0 + np.exp(-(c - 3.0))))))
    # Leaky variants. In a network the negative slope exists to keep gradients
    # alive; there are no gradients here, so the shape carries over but the
    # reason does not. What it means for us is concrete: below the threshold a
    # cheap episode gets a small DISCOUNT rather than merely no penalty, so the
    # transform pulls cheap promotions in as well as pushing costly ones out.
    for t, b, a in ((2.0, 1.0, 0.15), (4.0, 2.0, 0.10)):
        out.append((f"leaky t={t} b={b} a={a}",
                    lambda c, t=t, b=b, a=a: np.maximum(
                        c * 0.05,
                        c + b * np.where(c >= t, c - t, a * (c - t)))))
    return out


def concentration(ps, pc, prc, rc, budget, cfg):
    choice = allocate(ps, pc, prc, budget_multiplier=budget, **cfg)
    n = np.arange(len(choice))
    excess = rc[n, choice] - rc[:, 0]
    total = excess.sum()
    return (excess.max() / total if total > 0 else 0.0,
            rc[n, choice].sum() / rc[:, 0].sum())


def main() -> int:
    splits, data = load_splits(REPO_ROOT / "build/v20-matrices.npz")
    budgets = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))
    params = json.loads((REPO_ROOT / "build/v20-target-0.003.json").read_text(encoding="utf-8"))
    weights = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}

    print("랭킹용 비용만 변형 (예산 회계는 선형 유지). 등급별 safety는 재탐색.\n")
    print(f"{'형태':20}" + "".join(f"{t:>11}" for t in TIER_ORDER)
          + f"{'가중합':>10}{'최악상한':>10}{'fast집중':>10}")
    for label, transform in shapes():
        # normalize so a transform's scale does not silently change lambda's range
        reshaped = []
        for ps, pc, prc, rs, rc in splits:
            unit = prc[:, 0].mean()
            reshaped.append((ps, pc, transform(prc / unit) * unit, rs, rc))
        total, cells, worst_bound = 0.0, "", 0.0
        fast_conc = None
        for tier in TIER_ORDER:
            p = params[tier]
            risk = np.array([1.0, p["risk_mid"], p["risk_high"]])
            best = None
            for s in SAFETY_GRID:
                cfg = dict(safety_ratio=s, risk_multiplier=risk,
                           high_cap_ratio=p.get("high_cap_ratio", 1.0),
                           share_ratio=p.get("share_ratio", 1.0))
                rng = np.random.default_rng((13, TIER_ORDER.index(tier), int(s * 100)))
                _, score, counts, k = bootstrap(
                    reshaped, budget_multiplier=budgets[tier], rng=rng, k=1200,
                    with_counts=True, **cfg)
                bound = max(cp(int(c), k) for c in counts)
                if bound <= 0.005 and (best is None or score > best[0]):
                    best = (score, bound, cfg)
            if best is None:
                cells += f"{'실패':>11}"
                continue
            cells += f"{best[0]:>11.4f}"
            total += weights[tier] * best[0]
            worst_bound = max(worst_bound, best[1])
            if tier == "fast":
                fast_conc = concentration(*reshaped[0][:3], reshaped[0][4],
                                          budgets[tier], best[2])[0]
        print(f"{label:20}{cells}{total:>10.4f}{worst_bound:>9.2%}"
              f"{fast_conc:>9.1%}" if fast_conc is not None else
              f"{label:20}{cells}{total:>10.4f}{worst_bound:>9.2%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
