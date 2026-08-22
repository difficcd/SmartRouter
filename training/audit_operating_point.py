# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Certify one already-chosen configuration, at a k the search cannot afford.

The search bootstraps thousands of candidates, so it has to keep k small and
then picks the maximum -- which biases the winner's overrun downward. Auditing
a single fixed operating point has neither problem: one point means k can be
large, and nothing is being selected, so the observed rate is unbiased and the
Clopper-Pearson bound is a real certificate rather than a correction.

Used to answer "is what is on main actually provably safe", and to check any
v16 candidate on the same footing before it replaces it.

    py -3.13 training/audit_operating_point.py --matrices build/search-matrices.npz \
        --config v13 -k 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finalize_search import clopper_pearson_upper  # noqa: E402
from search_standalone import TIER_ORDER, bootstrap, load_splits  # noqa: E402

# v13 (currently on main) predates the safety_ratio reparameterization: its cap
# was light_total * max(1.0, budget_multiplier * safety_ratio). Expressed as a
# share of the tier's allowed excess -- what safety_ratio means now -- the same
# caps are these. Verified by cap/light: 1.05 / 1.60 / 2.40.
V13 = {
    "fast":     dict(safety_ratio=0.20,     risk_mid=2.5, risk_high=5.0),
    "balanced": dict(safety_ratio=0.60,     risk_mid=2.5, risk_high=2.0),
    "premium":  dict(safety_ratio=7.0 / 15, risk_mid=2.5, risk_high=4.0),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrices", required=True)
    parser.add_argument("--config", default="v13", help="v13, 또는 search 결과 JSON 경로")
    parser.add_argument("-k", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args(argv)

    if args.config == "v13":
        config = V13
    else:
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        config = {
            tier: dict(
                safety_ratio=raw[tier]["safety_ratio"],
                risk_mid=raw[tier]["risk_mid"],
                risk_high=raw[tier]["risk_high"],
                high_cap_ratio=raw[tier].get("high_cap_ratio", 1.0),
                share_ratio=raw[tier].get("share_ratio", 1.0),
            )
            for tier in TIER_ORDER
        }

    splits, data = load_splits(args.matrices)
    budget_multipliers = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))

    print(f"{args.config} 감사  k={args.k}  alpha={args.alpha}\n")
    print(f"{'등급':10}{'cap/light':>10}{'dev':>9}{'train':>9}{'95% 상한':>11}{'점수':>9}")
    worst_bound = 0.0
    for tier in TIER_ORDER:
        cfg = config[tier]
        bm = budget_multipliers[tier]
        rng = np.random.default_rng((args.seed, TIER_ORDER.index(tier)))
        _, score, counts, k = bootstrap(
            splits, budget_multiplier=bm,
            safety_ratio=cfg["safety_ratio"],
            risk_multiplier=np.array([1.0, cfg["risk_mid"], cfg["risk_high"]]),
            high_cap_ratio=cfg.get("high_cap_ratio", 1.0),
            share_ratio=cfg.get("share_ratio", 1.0),
            rng=rng, k=args.k, with_counts=True,
        )
        bound = max(clopper_pearson_upper(int(c), k, args.alpha) for c in counts)
        worst_bound = max(worst_bound, bound)
        cap = 1.0 + (bm - 1.0) * cfg["safety_ratio"]
        print(f"{tier:10}{cap:>10.3f}{counts[0]:>9}{counts[1]:>9}"
              f"{bound:>10.3%}{score:>9.4f}")
    print(f"\n최악 등급 95% 상한: {worst_bound:.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
