# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""How wrong can the world be before a tier's budget breaks?

Every safety number we have comes from bootstrapping the two public splits,
and a bootstrap can only replay variation that was actually observed. What we
observed is thin exactly where it matters: Dev contains one episode whose ax31
cost came out 40.4x its light cost (batch average 2.10x), and Train contains
none that extreme. So "how bad can a blowup be" and "how many can show up at
once" are questions our 2640 public episodes do not answer.

This asks them directly by editing the real cost matrix and re-running the
shipped allocator:

  scale     the observed worst blowup gets worse (40x -> 60x, 80x, ...)
  count     more episodes blow up (1 -> 2, 3, 5, 10)
  drift     the whole batch multiple moves outside the observed range
            (ax31 2.10/2.16 -> 2.5, 3.0; axk1 23.8/23.2 -> 28, 33)

The output is the margin in each dimension: the point where the tier stops
passing. That turns "we do not know the tail" into a number.

    py -3.13 training/stress_scenarios.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

from search_standalone import TIER_ORDER, allocate, load_splits  # noqa: E402


def search_params(path):
    """Parameters from a finalize_search result, so an arm that is not the one
    currently baked into heuristic.py can still be stressed. Comparing two arms
    otherwise means switching branches between runs, which is how the wrong
    numbers end up side by side in a table."""
    import json
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from ossp_router.protocol import MODEL_IDS
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        tier: dict(
            safety_ratio=float(r[tier]["safety_ratio"]),
            risk=np.array([1.0, float(r[tier]["risk_mid"]),
                           float(r[tier]["risk_high"])]),
            high_cap_ratio=float(r[tier]["high_cap_ratio"]),
            share_ratio=float(r[tier].get("share_ratio", 1.0)),
        )
        for tier in TIER_ORDER
    }


def artifact_params():
    import json
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from ossp_router.heuristic import _TEAM_ARTIFACT_JSON
    from ossp_router.protocol import MODEL_IDS
    a = json.loads(_TEAM_ARTIFACT_JSON)
    return {
        tier: dict(
            safety_ratio=float(a["tier_safety_ratios"][tier]),
            risk=np.array([float(a["risk_multiplier"][tier][m]) for m in MODEL_IDS]),
            high_cap_ratio=float(a["high_cap_ratio"][tier]),
            share_ratio=float(a.get("share_ratio", {}).get(tier, 1.0)),
        )
        for tier in TIER_ORDER
    }


def ratio(tier, params, budget, ps, pc, prc, rc):
    """Realized budget ratio when the router meets this (possibly edited) world.

    Predictions stay untouched on purpose -- the router cannot see the damage
    coming, which is the whole point.
    """
    cfg = params[tier]
    choice = allocate(ps, pc, prc, budget_multiplier=budget,
                      safety_ratio=cfg["safety_ratio"], risk_multiplier=cfg["risk"],
                      high_cap_ratio=cfg["high_cap_ratio"], share_ratio=cfg["share_ratio"])
    n = np.arange(len(choice))
    return rc[n, choice].sum() / rc[:, 0].sum()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrices", default=str(REPO_ROOT / "build/search-matrices.npz"))
    parser.add_argument("--params", default=None,
                        help="finalize_search 결과 JSON. 없으면 구워진 아티팩트를 씁니다")
    args = parser.parse_args(argv)

    splits, data = load_splits(args.matrices)
    budgets = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))
    params = search_params(args.params) if args.params else artifact_params()
    print(f"파라미터 출처: {args.params or '구워진 아티팩트'}")

    for split_name, (ps, pc, prc, rs, rc) in zip(("dev", "train"), splits):
        print(f"\n{'=' * 68}\n{split_name}")
        base = {t: ratio(t, params, budgets[t], ps, pc, prc, rc) for t in TIER_ORDER}
        print("기준 예산비율: " + "  ".join(
            f"{t}={base[t]:.4f}/{budgets[t]:.2f}" for t in TIER_ORDER))

        # 1) the worst observed blowup gets worse
        print("\n[1] 관측된 최악 폭주를 키웠을 때 (해당 문항의 ax31 배수를 강제)")
        print(f"{'ax31 배수':>10}" + "".join(f"{t:>14}" for t in TIER_ORDER))
        worst = int(np.argmax(rc[:, 1] - rc[:, 0]))
        for multiple in (40.4, 60.0, 80.0, 120.0, 200.0):
            damaged = rc.copy()
            damaged[worst, 1] = damaged[worst, 0] * multiple
            cells = ""
            for t in TIER_ORDER:
                r = ratio(t, params, budgets[t], ps, pc, prc, damaged)
                cells += f"{r:>11.4f}{'  !' if r > budgets[t] else '   '}"
            print(f"{multiple:>10.0f}{cells}")

        # 2) more episodes blow up. Promote the ones the router already picked,
        #    since an unpromoted blowup costs nothing.
        print("\n[2] 폭주 문항 개수를 늘렸을 때 (각각 ax31 40배)")
        print(f"{'개수':>10}" + "".join(f"{t:>14}" for t in TIER_ORDER))
        order = np.argsort(-pc[:, 0])  # biggest predicted light cost first
        for count in (1, 2, 3, 5, 10):
            damaged = rc.copy()
            damaged[order[:count], 1] = damaged[order[:count], 0] * 40.0
            cells = ""
            for t in TIER_ORDER:
                r = ratio(t, params, budgets[t], ps, pc, prc, damaged)
                cells += f"{r:>11.4f}{'  !' if r > budgets[t] else '   '}"
            print(f"{count:>10}{cells}")

        # 3) the batch-level multiple itself moves. This is the quantity the
        #    share-based bound assumes is stable (2.10 vs 2.16 across splits).
        print("\n[3] 배치 전체 비용배수가 관측 범위를 벗어났을 때")
        print(f"{'ax31':>6}{'axk1':>6}" + "".join(f"{t:>14}" for t in TIER_ORDER))
        base31 = rc[:, 1].sum() / rc[:, 0].sum()
        basek1 = rc[:, 2].sum() / rc[:, 0].sum()
        for m31, mk1 in ((base31, basek1), (2.5, 26.0), (3.0, 30.0), (4.0, 40.0), (6.0, 60.0)):
            damaged = rc.copy()
            damaged[:, 1] *= m31 / base31
            damaged[:, 2] *= mk1 / basek1
            cells = ""
            for t in TIER_ORDER:
                r = ratio(t, params, budgets[t], ps, pc, prc, damaged)
                cells += f"{r:>11.4f}{'  !' if r > budgets[t] else '   '}"
            print(f"{m31:>6.1f}{mk1:>6.1f}{cells}")
    print("\n! = 해당 등급 예산 초과 (0점 처리)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
