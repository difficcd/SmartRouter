# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Portable copy of the safety search -- numpy only, no repo imports.

Same algorithm as calibrate_safety.py's search, but reading the exported
matrices from export_matrices.py instead of loading the router. That makes it
runnable on any machine with numpy (including a free cloud notebook) so the
expensive bootstrap doesn't have to cook a laptop, and lets the per-tier
searches -- which are completely independent -- be split across machines.

    # locally, one tier, gentle:
    py -3.13 training/search_standalone.py --matrices build/search-matrices.npz \
        --tiers fast --workers 2 --out build/search-fast.json

    # elsewhere, the heavy two:
    python search_standalone.py --matrices search-matrices.npz \
        --tiers balanced premium --workers 2 --out search-rest.json

Merge the resulting JSONs with apply_search.py.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RNG_SEED = 20260815
BOOTSTRAP_K = 300
OVERRUN_TARGET = 0.01
SCORE_TOLERANCE = 0.001
SAFETY_GRID = [
    0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
    0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00,
]
RISK_HIGH_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
RISK_MID_GRID = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
HIGH_CAP_GRID = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50]
TIER_ORDER = ["fast", "balanced", "premium"]


def allocate(pred_scores, pred_costs, pred_rank_costs, *, budget_multiplier,
             safety_ratio, risk_multiplier, high_cap_ratio=1.0, iterations=60):
    light_total = pred_costs[:, 0].sum()
    rank_light_total = pred_rank_costs[:, 0].sum()
    cap = light_total * max(1.0, budget_multiplier * safety_ratio)
    high_cap = cap * high_cap_ratio

    def choose(penalty):
        ev = pred_scores - penalty * risk_multiplier[None, :] * pred_rank_costs / rank_light_total
        choice = np.argmax(ev, axis=1)
        total = pred_costs[np.arange(len(choice)), choice].sum()
        high_total = pred_costs[choice == 2, 2].sum()
        return choice, total, high_total

    def violates(total, high_total):
        return total > cap or high_total > high_cap

    choice, total, high_total = choose(0.0)
    if violates(total, high_total):
        low, high = 0.0, 1.0
        choice, total, high_total = choose(high)
        while violates(total, high_total) and high < 2.0 ** 40:
            low, high = high, high * 2.0
            choice, total, high_total = choose(high)
        for _ in range(iterations):
            middle = (low + high) / 2.0
            c, t, h = choose(middle)
            if not violates(t, h):
                high = middle
                choice, total, high_total = c, t, h
            else:
                low = middle
    if violates(total, high_total):
        choice = np.zeros(len(pred_scores), dtype=int)
    return choice


def bootstrap(splits, *, budget_multiplier, safety_ratio, risk_multiplier,
              high_cap_ratio, rng, k=BOOTSTRAP_K):
    """Worst overrun probability across splits, mean score across splits."""
    worst = 0.0
    scores = []
    for ps, pc, prc, rs, rc in splits:
        n = len(ps)
        overruns = 0
        total_score = 0.0
        for _ in range(k):
            idx = rng.integers(0, n, size=n)
            choice = allocate(
                ps[idx], pc[idx], prc[idx],
                budget_multiplier=budget_multiplier,
                safety_ratio=safety_ratio,
                risk_multiplier=risk_multiplier,
                high_cap_ratio=high_cap_ratio,
            )
            real_total = rc[idx, choice].sum()
            limit = rc[idx, 0].sum() * budget_multiplier
            if real_total > limit:
                overruns += 1
            else:
                total_score += float(rs[idx, choice].mean())
        worst = max(worst, overruns / k)
        scores.append(total_score / k)
    return worst, float(np.mean(scores))


_SPLITS = None


def _init(splits):
    global _SPLITS
    _SPLITS = splits


def _task(job):
    tier, budget_multiplier, risk_mid, risk_high = job
    rng = np.random.default_rng(
        (RNG_SEED, TIER_ORDER.index(tier), int(risk_mid * 100), int(risk_high * 100))
    )
    risk_multiplier = np.array([1.0, risk_mid, risk_high])
    best = None
    fallback = None
    for safety_ratio in SAFETY_GRID:
        overrun, score = bootstrap(
            _SPLITS, budget_multiplier=budget_multiplier, safety_ratio=safety_ratio,
            risk_multiplier=risk_multiplier, high_cap_ratio=1.0, rng=rng,
        )
        if fallback is None or overrun < fallback[2]:
            fallback = (score, safety_ratio, overrun)
        if overrun <= OVERRUN_TARGET and (best is None or score > best[0]):
            best = (score, safety_ratio, overrun)
    score, safety_ratio, overrun = best if best is not None else fallback
    return tier, risk_mid, risk_high, safety_ratio, overrun, score


def load_splits(path):
    data = np.load(path, allow_pickle=False)
    names = ("pred_scores", "pred_costs", "pred_rank_costs", "real_scores", "real_costs")
    return [tuple(data[f"{s}__{n}"] for n in names) for s in ("dev", "train")], data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrices", required=True)
    parser.add_argument("--tiers", nargs="+", default=TIER_ORDER)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    splits, data = load_splits(args.matrices)
    budget_multipliers = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))

    jobs = [
        (tier, budget_multipliers[tier], risk_mid, risk_high)
        for tier in args.tiers
        for risk_high in RISK_HIGH_GRID
        for risk_mid in RISK_MID_GRID
    ]
    print(f"등급 {args.tiers}: {len(jobs)}개 조합, 프로세스 {args.workers}개, K={BOOTSTRAP_K}")
    print(f"Dev {len(splits[0][0])}문항 / Train {len(splits[1][0])}문항, 두 split 모두 안전해야 채택\n")

    best = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(splits,)) as pool:
        for tier, risk_mid, risk_high, safety, overrun, score in pool.map(_task, jobs):
            done += 1
            if done % 12 == 0:
                print(f"  ... {done}/{len(jobs)}", flush=True)
            if tier not in best or score > best[tier]["score"]:
                best[tier] = {
                    "risk_mid": risk_mid, "risk_high": risk_high,
                    "safety_ratio": safety, "overrun": overrun, "score": score,
                }

    # Tighten the hard axk1 cap where it's free, now that the rest is fixed.
    for tier in args.tiers:
        b = best[tier]
        rng = np.random.default_rng((RNG_SEED, 999, TIER_ORDER.index(tier)))
        risk_multiplier = np.array([1.0, b["risk_mid"], b["risk_high"]])
        chosen = 1.0
        for high_cap_ratio in sorted(HIGH_CAP_GRID):
            overrun, score = bootstrap(
                splits, budget_multiplier=budget_multipliers[tier],
                safety_ratio=b["safety_ratio"], risk_multiplier=risk_multiplier,
                high_cap_ratio=high_cap_ratio, rng=rng,
            )
            if overrun <= OVERRUN_TARGET and score >= b["score"] - SCORE_TOLERANCE:
                chosen = high_cap_ratio
                break
        b["high_cap_ratio"] = chosen
        print(
            f"{tier:9} risk[ax31]={b['risk_mid']:.2f} risk[axk1]={b['risk_high']:.2f} "
            f"safety={b['safety_ratio']:.2f} high_cap={chosen:.2f} "
            f"최악초과율={b['overrun']:.3f} 점수={b['score']:.4f}"
        )

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(best, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\nOK: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
