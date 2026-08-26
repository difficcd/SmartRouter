# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Pick a safety point off a saved search curve, at any overrun target.

search_standalone.py --curve-out records the bootstrap (overrun, score) curve
over the whole SAFETY_GRID for every (tier, risk_mid, risk_high) combination.
That grid is where all the compute goes, and it does not depend on the overrun
target -- the target only decides which already measured point wins. So this
script replays the selection at whatever target we want and finishes the two
cheap refinement steps (high_cap_ratio, share_ratio), producing exactly the
JSON apply_search.py consumes.

Written for v16: v15 chased score at a 1% overrun target and won ~nothing while
Premium's safety margin fell from 101x the observed Dev->Train drift to 6.4x.
Being able to re-price that trade at 0.5% and 0.3% without re-running the
search is the whole point.

    py -3.13 training/finalize_search.py --curves build/curve-*.json \
        --matrices build/search-matrices.npz --overrun-target 0.005 \
        --out build/search-v16-005.json
"""

from __future__ import annotations

import argparse
import json
from math import exp, lgamma, log
from pathlib import Path

import numpy as np

from search_standalone import (  # noqa: E402
    HIGH_CAP_GRID,
    RNG_SEED,
    SCORE_TOLERANCE,
    SHARE_RATIO_GRID,
    TIER_ORDER,
    bootstrap,
    load_splits,
    allocate,
    pick_from_curve,
)


def _tier_ratio(tier, cfg, budget, split):
    """Budget ratio this configuration realises on one split.

    The curves record score and overrun but not what fraction of the cap was
    actually spent, so a tie-break on margin has to recompute it."""
    pred_scores, pred_costs, pred_rank_costs, _real_scores, real_costs = split
    choice = allocate(
        pred_scores, pred_costs, pred_rank_costs,
        budget_multiplier=budget,
        safety_ratio=cfg["safety_ratio"],
        risk_multiplier=np.array([1.0, cfg["risk_mid"], cfg["risk_high"]]),
        high_cap_ratio=cfg.get("high_cap_ratio", HIGH_CAP_GRID[0]),
        share_ratio=cfg.get("share_ratio", SHARE_RATIO_GRID[-1]),
    )
    idx = np.arange(len(choice))
    return float(real_costs[idx, choice].sum() / real_costs[:, 0].sum())


def _binom_cdf(x: int, n: int, p: float) -> float:
    """P(X <= x) for X ~ Binomial(n, p). x is tiny here, so summing is fine."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if x >= n else 0.0
    log_p, log_q = log(p), log(1.0 - p)
    return sum(
        exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log_p + (n - i) * log_q)
        for i in range(x + 1)
    )


def clopper_pearson_upper(x: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper confidence bound on a binomial rate.

    Why this and not the observed frequency: the search picks the highest
    scoring point out of ~2700 bootstrap estimates, so whichever point happened
    to draw an unluckily *low* overrun count is the one most likely to win.
    That is the same selection bias that made v15 look safe at a 1% target and
    then land with only 6.4x the observed Dev->Train drift as margin. Testing
    the upper bound instead means a point can only win by being genuinely safe,
    not by being lucky -- at k=1000 an observed 0/1000 still bounds at 0.30%,
    so the rule stays usable rather than rejecting everything.
    """
    if x >= n:
        return 1.0
    low, high = x / n, 1.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if _binom_cdf(x, n, mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def make_measure(rule: str, alpha: float):
    """Turn a curve row into the overrun figure the target is tested against."""
    if rule == "point":
        return None
    def measure(row):
        # rows are [safety, overrun, score, counts_per_split, k]
        if len(row) < 5:
            raise SystemExit(
                "이 곡선 파일에는 초과 횟수가 없다 (K=300 시절 형식). "
                "--rule point 로 돌리거나 곡선을 다시 만들어야 한다."
            )
        counts, k = row[3], row[4]
        return max(clopper_pearson_upper(int(c), int(k), alpha) for c in counts)
    return measure


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--curves", type=Path, nargs="+", required=True)
    parser.add_argument("--matrices", required=True)
    parser.add_argument("--overrun-target", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--rule", choices=("point", "ucb"), default="ucb",
        help="point=부트스트랩 빈도 그대로, ucb=Clopper-Pearson 상한 (기본).",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--score-tolerance", type=float, default=SCORE_TOLERANCE,
        help="high_cap/share 정밀화가 내줘도 되는 부트스트랩 점수 (기본 %(default)s). "
             "두 단계에 각각 적용되므로 실제 손실은 최대 2배가 될 수 있다. v16 실측: "
             "0.006으로 두자 fast가 한도를 다 써서 실측 점수를 0.014 잃었다 -- "
             "정밀화는 공짜일 때만 걸어야 하는 백스톱이므로 작게 잡는 편이 맞다.",
    )
    parser.add_argument(
        "--skip-refine", action="store_true",
        help="high_cap/share 정밀화를 건너뛰고 1.0으로 둔다 (빠른 비교용).",
    )
    args = parser.parse_args(argv)

    rows = []
    budget_multipliers = {}
    for path in args.curves:
        chunk = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(chunk["rows"])
        budget_multipliers.update(chunk["budget_multipliers"])

    measure = make_measure(args.rule, args.alpha)
    splits, _ = load_splits(args.matrices)

    # Keep every candidate. One episode can move a tier score by 1/n, so two
    # points whose scores differ by far less than that are tied in every sense
    # that matters -- and picking between them on score alone hands the
    # decision to bootstrap noise. v22's premium was chosen over a point
    # scoring 1.2e-5 less that held 2.7x less variation between the splits.
    candidates = {}
    for row in rows:
        safety, overrun, score = pick_from_curve(
            row["curve"], args.overrun_target, measure)
        candidates.setdefault(row["tier"], []).append({
            "risk_mid": row["risk_mid"], "risk_high": row["risk_high"],
            "safety_ratio": safety, "overrun": overrun, "score": score,
        })

    best = {}
    for tier, cands in candidates.items():
        top = max(c["score"] for c in cands)
        n = len(splits[0][0])
        # A tenth of what one episode can move the score. One full episode is
        # too loose a definition of "tied": at that width the better-margin pick
        # can cost real score, and a difference that large is not obviously
        # noise. A tenth is unambiguous -- no single observation explains it.
        tol = 0.1 / n
        tied = [c for c in cands if top - c["score"] <= tol]
        if len(tied) == 1:
            best[tier] = tied[0]
            continue
        # Among the tied, take the one whose worst split leaves most room.
        scored = []
        for c in tied:
            ratios = [_tier_ratio(tier, c, budget_multipliers[tier], s)
                      for s in splits]
            scored.append((budget_multipliers[tier] - max(ratios), c, ratios))
        scored.sort(key=lambda x: -x[0])
        room, chosen, ratios = scored[0]
        best[tier] = chosen
        if chosen is not max(tied, key=lambda c: c["score"]):
            other = max(tied, key=lambda c: c["score"])
            print(f"  {tier}: 동점 {len(tied)}개 중 여유가 큰 쪽 채택 "
                  f"(점수 {chosen['score']:.6f} vs {other['score']:.6f}, "
                  f"차이 {other['score']-chosen['score']:.2e} "
                  f"< 1문항의 1/10 {tol:.2e})")


    missing = [t for t in TIER_ORDER if t not in best]
    if missing:
        raise SystemExit(f"곡선에 없는 등급: {missing}")

    rule_note = "Clopper-Pearson 상한" if args.rule == "ucb" else "부트스트랩 빈도"
    if args.rule == "ucb":
        rule_note += f", alpha={args.alpha}"
    print(f"초과확률 목표 {args.overrun_target:.3%} 판정={rule_note}  "
          f"({len(rows)}개 조합의 곡선에서 선택)\n")

    for tier in TIER_ORDER:
        b = best[tier]
        b["high_cap_ratio"] = 1.0
        b["share_ratio"] = 1.0
        if not args.skip_refine:
            rng = np.random.default_rng((RNG_SEED, 999, TIER_ORDER.index(tier)))
            risk_multiplier = np.array([1.0, b["risk_mid"], b["risk_high"]])
            kwargs = dict(
                budget_multiplier=budget_multipliers[tier],
                safety_ratio=b["safety_ratio"], risk_multiplier=risk_multiplier,
            )
            def judged(**extra):
                """(overrun figure under the active rule, score)."""
                frequency, score, counts, k = bootstrap(
                    splits, rng=rng, with_counts=True, **kwargs, **extra)
                row = [None, frequency, score, counts, k]
                return (measure(row) if measure else frequency), score

            for high_cap_ratio in sorted(HIGH_CAP_GRID):
                overrun, score = judged(high_cap_ratio=high_cap_ratio)
                if overrun <= args.overrun_target and score >= b["score"] - args.score_tolerance:
                    b["high_cap_ratio"] = high_cap_ratio
                    break
            for share_ratio in sorted(SHARE_RATIO_GRID):
                overrun, score = judged(
                    high_cap_ratio=b["high_cap_ratio"], share_ratio=share_ratio)
                if overrun <= args.overrun_target and score >= b["score"] - args.score_tolerance:
                    b["share_ratio"] = share_ratio
                    break
        print(
            f"{tier:9} risk[ax31]={b['risk_mid']:.2f} risk[axk1]={b['risk_high']:.2f} "
            f"safety={b['safety_ratio']:.2f} high_cap={b['high_cap_ratio']:.2f} "
            f"share={b['share_ratio']:.2f} "
            f"최악초과율={b['overrun']:.3f} 점수={b['score']:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(best, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nOK: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
