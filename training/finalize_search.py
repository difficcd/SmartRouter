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
    pick_from_curve,
)


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
    best = {}
    for row in rows:
        safety, overrun, score = pick_from_curve(
            row["curve"], args.overrun_target, measure)
        tier = row["tier"]
        if tier not in best or score > best[tier]["score"]:
            best[tier] = {
                "risk_mid": row["risk_mid"], "risk_high": row["risk_high"],
                "safety_ratio": safety, "overrun": overrun, "score": score,
            }

    missing = [t for t in TIER_ORDER if t not in best]
    if missing:
        raise SystemExit(f"곡선에 없는 등급: {missing}")

    splits, _ = load_splits(args.matrices)
    print(f"초과확률 목표 {args.overrun_target:.3%}  ({len(rows)}개 조합의 곡선에서 선택)\n")

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
                if overrun <= args.overrun_target and score >= b["score"] - SCORE_TOLERANCE:
                    b["high_cap_ratio"] = high_cap_ratio
                    break
            for share_ratio in sorted(SHARE_RATIO_GRID):
                overrun, score = judged(
                    high_cap_ratio=b["high_cap_ratio"], share_ratio=share_ratio)
                if overrun <= args.overrun_target and score >= b["score"] - SCORE_TOLERANCE:
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
