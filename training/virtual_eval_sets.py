# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Route many randomly-composed evaluation sets and count the budget failures.

The other two stress tools each fix one thing and vary another. private_set_stress
walks a handful of deliberately skewed compositions; stress_scenarios holds the
composition and inflates cost. Both are small and deterministic, which makes them
easy to read and easy to satisfy by accident.

This one varies size, composition and cost together, at random, many times. The
question it answers is the one the organisers' warning actually poses -- the
private set is drawn differently, and we do not get to know how -- so the useful
output is a failure count over hundreds of draws rather than a verdict on three.

Each virtual set draws a size, then a skew axis and strength, then optionally
shifts cost, then routes all three tiers and records whether each stayed inside
its cap.

    py -3.13 training/virtual_eval_sets.py --params build/v22e-params.json \
        --matrices training/tmp-v22-matrices.npz --sets 300 --seeds 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

from search_standalone import TIER_ORDER, allocate, load_splits  # noqa: E402

WEIGHTS = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}


def pooled(splits):
    """Both public splits as one pool -- 2,640 episodes to draw from."""
    keys = ("pred_scores", "pred_costs", "pred_rank_costs", "real_scores", "real_costs")
    return tuple(np.concatenate([s[i] for s in splits], axis=0) for i in range(len(keys)))


def axes(pool):
    """Orderings a private set could plausibly be skewed along."""
    ps, pc, prc, rs, rc = pool
    light = rc[:, 0]
    return {
        "light 비용": light,
        "axk1-light 비용비": rc[:, 2] / np.maximum(light, 1e-12),
        "ax31 승격 이득": rs[:, 1] - rs[:, 0],
        "light 점수": rs[:, 0],
    }


def draw(rng, pool, axis_values, n):
    """One virtual set: pick a skew axis, a strength, and a cost shift."""
    total = len(axis_values)
    strength = rng.choice([0.0, 0.3, 0.5, 0.7, 1.0])
    if strength == 0.0:
        idx = rng.choice(total, size=n, replace=False)
    else:
        # Take the top fraction of one axis, fill the rest at random.
        order = np.argsort(-axis_values if rng.random() < 0.5 else axis_values)
        take = int(n * strength)
        head = order[:take]
        rest = np.setdiff1d(order, head, assume_unique=False)
        tail = rng.choice(rest, size=n - take, replace=False)
        idx = np.concatenate([head, tail])
    cost_shift = float(rng.choice([1.0, 1.0, 1.0, 1.1, 1.2, 1.3]))
    return idx, strength, cost_shift


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--matrices", required=True)
    parser.add_argument("--sets", type=int, default=300, help="시드당 가상셋 수")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--sizes", type=int, nargs="+", default=[600, 880, 1200])
    args = parser.parse_args(argv)

    cfg = json.loads(args.params.read_text(encoding="utf-8"))
    splits, data = load_splits(args.matrices)
    budgets = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))
    pool = pooled(splits)
    ps, pc, prc, rs, rc = pool
    axis_map = axes(pool)
    axis_names = list(axis_map)

    print(f"{args.params.name}  풀 {len(ps)}문항")
    print(f"가상셋 {args.sets} x 시드 {args.seeds} = {args.sets * args.seeds}회, "
          f"등급 3개 => {args.sets * args.seeds * 3}번 배분\n")

    failures = {t: 0 for t in TIER_ORDER}
    worst_ratio = {t: 0.0 for t in TIER_ORDER}
    total_zero = 0
    detail = []

    for seed in range(args.seeds):
        rng = np.random.default_rng(20260826 + seed)
        for _ in range(args.sets):
            n = int(rng.choice(args.sizes))
            axis = axis_names[int(rng.integers(len(axis_names)))]
            idx, strength, shift = draw(rng, pool, axis_map[axis], n)
            sub_rc = rc[idx].copy()
            if shift != 1.0:
                sub_rc[:, 1] *= shift
                sub_rc[:, 2] *= shift
            over = []
            for tier in TIER_ORDER:
                c = cfg[tier]
                choice = allocate(
                    ps[idx], pc[idx], prc[idx], budget_multiplier=budgets[tier],
                    safety_ratio=c["safety_ratio"],
                    risk_multiplier=np.array([1.0, c["risk_mid"], c["risk_high"]]),
                    high_cap_ratio=c["high_cap_ratio"], share_ratio=c["share_ratio"])
                k = np.arange(len(choice))
                ratio = sub_rc[k, choice].sum() / sub_rc[:, 0].sum()
                worst_ratio[tier] = max(worst_ratio[tier], ratio)
                if ratio > budgets[tier]:
                    failures[tier] += 1
                    over.append(tier)
            if len(over) == 3:
                total_zero += 1
            if over:
                detail.append((n, axis, strength, shift, tuple(over)))

    runs = args.sets * args.seeds
    print(f"{'등급':10s} {'한도':>6s} {'최악 비율':>10s} {'여유':>8s} {'초과':>12s}")
    for tier in TIER_ORDER:
        cap = budgets[tier]
        w = worst_ratio[tier]
        print(f"{tier:10s} {cap:6.2f} {w:10.4f} {(cap/w-1)*100:7.1f}% "
              f"{failures[tier]:6d}/{runs:5d}")
    print()
    print(f"세 등급 동시 초과 (총점 0): {total_zero}/{runs}")
    if detail:
        print(f"\n실패한 구성 {len(detail)}건 중 앞 10건:")
        for n, axis, strength, shift, over in detail[:10]:
            print(f"  크기 {n:5d}  축 {axis:16s} 치우침 {strength:.1f}  "
                  f"비용 x{shift:.1f}  초과 {','.join(over)}")
    else:
        print("\n실패 0건 -- 모든 가상셋에서 세 등급 모두 한도 안입니다.")
    return 1 if total_zero else 0


if __name__ == "__main__":
    raise SystemExit(main())
