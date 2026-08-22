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
# k=300 resolves the overrun probability only to 1/300 = 0.33%, so a 0.5%
# target degenerates into "at most one overrun in 300" -- mostly noise. v16
# tightens the target, so it needs the resolution to go with it: at k=1000 a
# 0.5% target is "at most 5 in 1000", and the Clopper-Pearson bound used by
# finalize_search.py's ucb rule becomes tight enough to be worth applying.
BOOTSTRAP_K = 1000
OVERRUN_TARGET = 0.01
# How much bootstrap score we will pay to buy a hard axk1 ceiling. At
# 0.001 the search never found a binding cap (v13 chose 1.0 = no cap for
# all three tiers), so the backstop we built has never actually engaged.
# Overrunning costs the whole tier, so a few thousandths is cheap here.
SCORE_TOLERANCE = 0.006
# safety_ratio is now "share of the tier's allowed excess" (see allocate's
# cap), so the whole 0-1 range is live for every tier -- the old fine steps
# clustered at 0.80-1.00 were an artifact of fast's dead zone below 0.80.
SAFETY_GRID = [
    # fast's usable band sits at the very bottom of this range: its excess is
    # only 0.25x, so safety 0.30 already means cap 1.075, which measured 3%
    # overrun -- above target. The old parameterization reached that region
    # through values below 0.80; this one needs the low end spelled out.
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.84, 0.88, 0.92, 0.96, 1.00,
]
RISK_HIGH_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
RISK_MID_GRID = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
HIGH_CAP_GRID = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50]
TIER_ORDER = ["fast", "balanced", "premium"]

# Deliberately loose upper bounds on each model's batch-level cost multiple
# relative to ax31-light, keyed by column index. Measured on the public data:
# ax31 2.1020 (Dev) / 2.1550 (Train); axk1-think 23.795 / 23.150. These bounds
# sit ~40% above the larger observation, so the share-based constraint in
# allocate() stays valid across a distribution shift many times larger than
# anything the two public splits show.
MODEL_MULTIPLE_BOUND = {1: 3.0, 2: 33.0}

# How much of the tier's allowed excess the share-based bound may claim.
# 1.0 means "the bound alone must fit inside the budget"; searched per tier.
SHARE_RATIO_GRID = [1.0, 0.9, 0.8, 0.7, 0.6]


def allocate(pred_scores, pred_costs, pred_rank_costs, *, budget_multiplier,
             safety_ratio, risk_multiplier, high_cap_ratio=1.0,
             share_ratio=1.0, iterations=60):
    light_total = pred_costs[:, 0].sum()
    rank_light_total = pred_rank_costs[:, 0].sum()
    # safety_ratio scales the ALLOWED EXCESS, not the whole multiplier.
    #
    # The old form (light_total * max(1.0, budget_multiplier * safety_ratio))
    # made safety_ratio mean wildly different things per tier: premium's 4.0x
    # budget at safety 0.7 still allowed 2.8x, but fast's 1.25x at safety 0.7
    # hit the max(1.0, ...) floor and allowed no promotion at all. Every
    # safety_ratio below 0.80 was the same dead value for fast, and 0.84 --
    # the value the search kept picking -- bought only 5% of the 25% it was
    # allowed to spend.
    #
    # Reading it as a fraction of the excess makes the parameter mean the same
    # thing everywhere ("spend this share of what the tier may spend beyond
    # all-light") and gives fast the same search resolution the other tiers
    # already had.
    cap = light_total * (1.0 + (budget_multiplier - 1.0) * safety_ratio)
    high_cap = cap * high_cap_ratio

    # Share-based hard constraint (see MODEL_MULTIPLE_BOUND).
    #
    # The budget check is really
    #     ratio = 1 + sum_promoted(c_m(i) - c_light(i)) / L
    # and grouping by model turns that into
    #     ratio = 1 + sum_m (rho_m - 1) * s_m
    # where s_m is the share of the batch's LIGHT cost held by the episodes
    # promoted to m, and rho_m is that model's batch-level cost multiple.
    #
    # This is worth doing because the two quantities have wildly different
    # reliability. Per-episode cost prediction correlates only 0.46-0.62 with
    # reality, but the batch-level multiple barely moves between splits:
    # ax31 2.1020 -> 2.1550 (+2.5%), axk1 23.795 -> 23.150 (+2.7%). And s_m is
    # a normalized share, so an error in the overall level of predicted light
    # cost cancels out of it.
    #
    # Bounding with a deliberately loose rho (MODEL_MULTIPLE_BOUND) therefore
    # gives a constraint that holds unless the batch multiple exceeds a value
    # far outside anything observed -- unlike safety_ratio and high_cap_ratio,
    # which both inherit the per-episode prediction's error.
    light_share = pred_costs[:, 0] / light_total
    excess_allowed = budget_multiplier - 1.0

    def choose(penalty):
        ev = pred_scores - penalty * risk_multiplier[None, :] * pred_rank_costs / rank_light_total
        choice = np.argmax(ev, axis=1)
        total = pred_costs[np.arange(len(choice)), choice].sum()
        high_total = pred_costs[choice == 2, 2].sum()
        bound_excess = sum(
            (MODEL_MULTIPLE_BOUND[m] - 1.0) * light_share[choice == m].sum()
            for m in (1, 2)
        )
        return choice, total, high_total, bound_excess

    def violates(total, high_total, bound_excess):
        return (
            total > cap
            or high_total > high_cap
            or bound_excess > excess_allowed * share_ratio
        )

    choice, total, high_total, bound_excess = choose(0.0)
    if violates(total, high_total, bound_excess):
        low, high = 0.0, 1.0
        choice, total, high_total, bound_excess = choose(high)
        while violates(total, high_total, bound_excess) and high < 2.0 ** 40:
            low, high = high, high * 2.0
            choice, total, high_total, bound_excess = choose(high)
        for _ in range(iterations):
            middle = (low + high) / 2.0
            c, t, h, b = choose(middle)
            if not violates(t, h, b):
                high = middle
                choice, total, high_total, bound_excess = c, t, h, b
            else:
                low = middle
    if violates(total, high_total, bound_excess):
        choice = np.zeros(len(pred_scores), dtype=int)
    return choice


def bootstrap(splits, *, budget_multiplier, safety_ratio, risk_multiplier,
              high_cap_ratio, share_ratio=1.0, rng, k=BOOTSTRAP_K,
              with_counts=False):
    """Worst overrun probability across splits, mean score across splits.

    with_counts additionally returns the raw (overruns, k) per split. The
    counts are what makes a confidence bound possible later: selecting the
    best of thousands of noisy overrun estimates favours the ones that got
    lucky, and a point estimate cannot express that. See finalize_search.py.
    """
    worst = 0.0
    scores = []
    counts = []
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
                share_ratio=share_ratio,
            )
            real_total = rc[idx, choice].sum()
            limit = rc[idx, 0].sum() * budget_multiplier
            if real_total > limit:
                overruns += 1
            else:
                total_score += float(rs[idx, choice].mean())
        worst = max(worst, overruns / k)
        scores.append(total_score / k)
        counts.append(overruns)
    if with_counts:
        return worst, float(np.mean(scores)), counts, k
    return worst, float(np.mean(scores))


_SPLITS = None


def _init(splits):
    global _SPLITS
    _SPLITS = splits


def _task(job):
    """Bootstrap the whole SAFETY_GRID and return the full curve.

    The expensive part -- k=300 resamples per (risk, safety) point -- does not
    depend on OVERRUN_TARGET at all; the target only decides which already
    computed point gets picked. Returning the curve instead of one winner
    therefore lets any target be evaluated afterwards for free, which is how
    v16 compares 1.0% / 0.5% / 0.3% off a single search.
    """
    tier, budget_multiplier, risk_mid, risk_high = job
    rng = np.random.default_rng(
        (RNG_SEED, TIER_ORDER.index(tier), int(risk_mid * 100), int(risk_high * 100))
    )
    risk_multiplier = np.array([1.0, risk_mid, risk_high])
    curve = []
    for safety_ratio in SAFETY_GRID:
        overrun, score, counts, k = bootstrap(
            _SPLITS, budget_multiplier=budget_multiplier, safety_ratio=safety_ratio,
            risk_multiplier=risk_multiplier, high_cap_ratio=1.0, rng=rng,
            with_counts=True,
        )
        curve.append([safety_ratio, overrun, score, counts, k])
    return tier, risk_mid, risk_high, curve


def pick_from_curve(curve, overrun_target, measure=None):
    """Highest-scoring safety_ratio meeting the target; else the safest one.

    measure(row) -> overrun figure to test against the target. The default is
    the raw bootstrap frequency; finalize_search.py passes a confidence bound
    instead. Returns (safety_ratio, measured_overrun, score).
    """
    if measure is None:
        measure = lambda row: row[1]  # noqa: E731
    scored = [(row[0], measure(row), row[2]) for row in curve]
    ok = [row for row in scored if row[1] <= overrun_target]
    if ok:
        return max(ok, key=lambda row: row[2])
    return min(scored, key=lambda row: row[1])


def load_splits(path):
    data = np.load(path, allow_pickle=False)
    names = ("pred_scores", "pred_costs", "pred_rank_costs", "real_scores", "real_costs")
    return [tuple(data[f"{s}__{n}"] for n in names) for s in ("dev", "train")], data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrices", required=True)
    parser.add_argument("--tiers", nargs="+", default=TIER_ORDER)
    parser.add_argument(
        "--risk-high",
        nargs="+",
        type=float,
        default=None,
        help="RISK_HIGH_GRID 중 이 값들만 탐색 (한 등급을 여러 대에 쪼갤 때). "
             "부분 결과들은 apply_search.py가 등급별 최고점으로 합친다.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--overrun-target", type=float, default=OVERRUN_TARGET,
        help="이 부트스트랩 초과확률 이하에서 점수 최대화 (기본 %(default)s).",
    )
    parser.add_argument(
        "--curve-out", default=None,
        help="safety_ratio 전 구간의 (초과확률, 점수) 곡선을 저장한다. "
             "탐색 비용은 목표값과 무관하므로, 이걸 남겨두면 다른 목표값을 "
             "재계산 없이 finalize_search.py로 고를 수 있다.",
    )
    args = parser.parse_args(argv)

    splits, data = load_splits(args.matrices)
    budget_multipliers = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))

    risk_high_grid = args.risk_high if args.risk_high else RISK_HIGH_GRID
    unknown = set(risk_high_grid) - set(RISK_HIGH_GRID)
    if unknown:
        parser.error(f"RISK_HIGH_GRID에 없는 값: {sorted(unknown)}")
    jobs = [
        (tier, budget_multipliers[tier], risk_mid, risk_high)
        for tier in args.tiers
        for risk_high in risk_high_grid
        for risk_mid in RISK_MID_GRID
    ]
    print(f"등급 {args.tiers} risk_high={risk_high_grid}: "
          f"{len(jobs)}개 조합, 프로세스 {args.workers}개, K={BOOTSTRAP_K}")
    print(f"Dev {len(splits[0][0])}문항 / Train {len(splits[1][0])}문항, 두 split 모두 안전해야 채택\n")

    best = {}
    curves = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(splits,)) as pool:
        for tier, risk_mid, risk_high, curve in pool.map(_task, jobs):
            done += 1
            if done % 12 == 0:
                print(f"  ... {done}/{len(jobs)}", flush=True)
            curves.append({
                "tier": tier, "risk_mid": risk_mid, "risk_high": risk_high,
                "curve": curve,
            })
            safety, overrun, score = pick_from_curve(curve, args.overrun_target)
            if tier not in best or score > best[tier]["score"]:
                best[tier] = {
                    "risk_mid": risk_mid, "risk_high": risk_high,
                    "safety_ratio": safety, "overrun": overrun, "score": score,
                }

    if args.curve_out:
        with open(args.curve_out, "w", encoding="utf-8") as handle:
            json.dump({"overrun_target": args.overrun_target,
                       "budget_multipliers": budget_multipliers,
                       "rows": curves}, handle, ensure_ascii=False)
        print(f"곡선 저장: {args.curve_out}")

    # Tighten the hard axk1 cap where it's free, now that the rest is fixed.
    # Only meaningful once a tier's whole grid has been seen -- with a partial
    # --risk-high slice, this machine's local winner may not be the tier's
    # actual winner, so leave the cap open and let apply_search.py decide after
    # the slices are merged.
    partial = args.risk_high is not None and set(risk_high_grid) != set(RISK_HIGH_GRID)
    if partial:
        for tier in args.tiers:
            best[tier]["high_cap_ratio"] = 1.0
            best[tier]["share_ratio"] = 1.0
            b = best[tier]
            print(
                f"{tier:9} (부분 탐색) risk[ax31]={b['risk_mid']:.2f} "
                f"risk[axk1]={b['risk_high']:.2f} safety={b['safety_ratio']:.2f} "
                f"최악초과율={b['overrun']:.3f} 점수={b['score']:.4f}"
            )
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(best, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"\nOK: {args.out} (부분 결과 -- 다른 조각과 합쳐야 함)")
        return 0

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
            if overrun <= args.overrun_target and score >= b["score"] - SCORE_TOLERANCE:
                chosen = high_cap_ratio
                break
        b["high_cap_ratio"] = chosen

        # Then tighten the share-based bound the same way. This one is the
        # prediction-independent guarantee, so it is worth more than the
        # per-episode caps above -- take the strictest value that still holds
        # the score.
        chosen_share = 1.0
        for share_ratio in sorted(SHARE_RATIO_GRID):
            overrun, score = bootstrap(
                splits, budget_multiplier=budget_multipliers[tier],
                safety_ratio=b["safety_ratio"], risk_multiplier=risk_multiplier,
                high_cap_ratio=chosen, share_ratio=share_ratio, rng=rng,
            )
            if overrun <= args.overrun_target and score >= b["score"] - SCORE_TOLERANCE:
                chosen_share = share_ratio
                break
        b["share_ratio"] = chosen_share

        print(
            f"{tier:9} risk[ax31]={b['risk_mid']:.2f} risk[axk1]={b['risk_high']:.2f} "
            f"safety={b['safety_ratio']:.2f} high_cap={chosen:.2f} "
            f"share={chosen_share:.2f} "
            f"최악초과율={b['overrun']:.3f} 점수={b['score']:.4f}"
        )

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(best, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\nOK: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
