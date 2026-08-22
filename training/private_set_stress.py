# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Estimate how the router behaves on a set it has never seen.

The private set decides everything, and the bootstrap cannot speak to it: it
resamples WITH REPLACEMENT from the very episodes we already have, so it
reproduces sampling noise around a distribution we fixed by observing it. A
genuinely new set can differ in composition, not just in draw.

Three views that the bootstrap does not give:

  disjoint   Cut the public pool into NON-OVERLAPPING chunks the size of a
             real split and run the router on each. Unlike a bootstrap sample,
             two chunks share no episode, so the spread between them is what
             independent samples actually do.

  composition  Rebuild a chunk with a deliberately skewed makeup -- mostly
             long prompts, mostly short, mostly code-like -- to ask what
             happens if the private set simply looks different.

  adversarial  Take the episodes this configuration handles worst and pack a
             whole split with them. Not a forecast; a floor.

Reported as realized budget ratio against the tier limit, because that is the
quantity that zeroes a tier.

    py -3.13 training/private_set_stress.py --params build/v19-params.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "training"))

from ossp_router.heuristic import _team_raw_feature_vector, episode_text  # noqa: E402
from ossp_router.protocol import load_input  # noqa: E402
from search_standalone import TIER_ORDER, allocate, load_splits  # noqa: E402


def characteristics():
    """Per-episode prompt length, pooled dev+train in the matrices' order."""
    out = []
    for split in ("dev", "train"):
        batch = load_input(REPO_ROOT / f"data/materialized/{split}/inputs.json")
        out.extend(len(episode_text(e)) for e in batch.episodes)
    return np.array(out, dtype=float)


def run(idx, tier, budget, cfg, pooled):
    ps, pc, prc, rs, rc = (a[idx] for a in pooled)
    choice = allocate(ps, pc, prc, budget_multiplier=budget, **cfg)
    n = np.arange(len(choice))
    return rc[n, choice].sum() / rc[:, 0].sum(), rs[n, choice].mean()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--matrices", default=str(REPO_ROOT / "build/search-matrices.npz"))
    parser.add_argument("--chunk", type=int, default=880, help="한 split 크기 (기본 %(default)s)")
    args = parser.parse_args(argv)

    splits, data = load_splits(args.matrices)
    budgets = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))
    pooled = tuple(np.concatenate([splits[0][i], splits[1][i]]) for i in range(5))
    total = len(pooled[0])
    chars = characteristics()
    params = json.loads(args.params.read_text(encoding="utf-8"))
    cfgs = {
        t: dict(safety_ratio=params[t]["safety_ratio"],
                risk_multiplier=np.array([1.0, params[t]["risk_mid"], params[t]["risk_high"]]),
                high_cap_ratio=params[t].get("high_cap_ratio", 1.0),
                share_ratio=params[t].get("share_ratio", 1.0))
        for t in TIER_ORDER
    }
    print(f"{args.params.name}   공개 {total}문항을 {args.chunk}문항 단위로 본다\n")

    # 1) disjoint chunks -- independent samples, not resamples
    print(f"[1] 겹치지 않는 {args.chunk}문항 조각 (복원추출 아님)")
    print(f"{'등급':10}{'한도':>7}{'최소':>9}{'중앙':>9}{'최대':>9}{'최악 여유':>11}  판정")
    rng = np.random.default_rng(20260822)
    order = rng.permutation(total)
    chunks = [order[i:i + args.chunk] for i in range(0, total - args.chunk + 1, args.chunk)]
    for tier in TIER_ORDER:
        vals = np.array([run(c, tier, budgets[tier], cfgs[tier], pooled)[0] for c in chunks])
        worst = vals.max()
        room = (budgets[tier] - worst) / worst
        print(f"{tier:10}{budgets[tier]:>7.2f}{vals.min():>9.4f}{np.median(vals):>9.4f}"
              f"{worst:>9.4f}{room:>10.1%}  {'통과' if worst <= budgets[tier] else '초과!!'}")

    # 2) composition shift -- the set simply looks different
    print(f"\n[2] 구성이 치우친 {args.chunk}문항 (길이 기준)")
    print(f"{'구성':22}" + "".join(f"{t:>12}" for t in TIER_ORDER))
    long_first = np.argsort(-chars)
    short_first = np.argsort(chars)
    scenarios = {
        "가장 긴 프롬프트만": long_first[:args.chunk],
        "가장 짧은 프롬프트만": short_first[:args.chunk],
        "긴 70% + 짧은 30%": np.concatenate([long_first[:int(args.chunk * 0.7)],
                                             short_first[:args.chunk - int(args.chunk * 0.7)]]),
    }
    for label, idx in scenarios.items():
        cells = ""
        for tier in TIER_ORDER:
            r, _ = run(idx, tier, budgets[tier], cfgs[tier], pooled)
            cells += f"{r:>10.4f}{'!!' if r > budgets[tier] else '  '}"
        print(f"{label:22}{cells}")

    # 3) adversarial -- pack a split with what this config handles worst
    print(f"\n[3] 이 설정이 가장 못 버티는 {args.chunk}문항만 모았을 때")
    print(f"{'등급':10}{'한도':>7}{'비율':>10}  판정")
    for tier in TIER_ORDER:
        ps, pc, prc, rs, rc = pooled
        choice = allocate(ps, pc, prc, budget_multiplier=budgets[tier], **cfgs[tier])
        n = np.arange(len(choice))
        excess = rc[n, choice] - rc[:, 0]
        idx = np.argsort(-excess)[:args.chunk]
        r, _ = run(idx, tier, budgets[tier], cfgs[tier], pooled)
        print(f"{tier:10}{budgets[tier]:>7.2f}{r:>10.4f}  {'통과' if r <= budgets[tier] else '초과!!'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
