# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Shrink the per-model score heads toward their shared component.

The three score heads are independent ridge fits on the same features, but the
thing the features actually carry -- how hard this prompt is -- is common to
all three models. Fitting it three times means estimating one signal three
times and adding three lots of noise to it.

What the router needs is the DIFFERENCE between heads, and we have measured
that difference to be nearly worthless for ax31: predicted ax31-over-light
gain correlates -0.05 with reality. axk1 is the opposite -- its vocabulary
signal is real and strong (def t=13.7, assert 13.4). So shrinking each head
toward the cross-model mean should kill a difference that is noise while
leaving one that is not, and the right amount of shrinkage differs per model.

    beta_m <- (1 - lam_m) * beta_m + lam_m * mean_over_models(beta)

No refit is needed to try it -- the heads are already linear, so this is a
closed-form edit, and the allocator can be re-run on the result directly.

    py -3.13 training/experiment_head_pooling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "training"))

from finalize_search import clopper_pearson_upper as cp  # noqa: E402
from ossp_router.heuristic import (  # noqa: E402
    _TEAM_ARTIFACT_JSON,
    _team_raw_feature_vector,
)
from ossp_router.protocol import MODEL_IDS, load_input  # noqa: E402
from search_standalone import SAFETY_GRID, TIER_ORDER, bootstrap, load_splits  # noqa: E402

RISK = {"fast": (1.0, 1.5), "balanced": (1.0, 2.5), "premium": (1.0, 6.0)}


def predicted_scores(artifact, lam):
    """Score matrix per split after shrinking heads toward their mean by lam."""
    coef = np.array([artifact["score_heads"][m]["coefficients"] for m in MODEL_IDS])
    inter = np.array([artifact["score_heads"][m]["intercept"] for m in MODEL_IDS])
    lam = np.asarray(lam, dtype=float).reshape(-1, 1)
    pooled = (1.0 - lam) * coef + lam * coef.mean(axis=0, keepdims=True)
    pooled_i = (1.0 - lam.ravel()) * inter + lam.ravel() * inter.mean()
    mean = np.array(artifact["feature_mean"])
    scale = np.array(artifact["feature_scale"])
    out = []
    for split in ("dev", "train"):
        batch = load_input(REPO_ROOT / f"data/materialized/{split}/inputs.json")
        X = np.array([_team_raw_feature_vector(e, artifact["hash_bins"])
                      for e in batch.episodes], dtype=float)
        Z = (X - mean) / scale
        out.append(np.clip(Z @ pooled.T + pooled_i, 0.0, 1.0))
    return out


def best_at_bound(splits, tier, budget, bound=0.003, k=2000):
    best = None
    for s in SAFETY_GRID:
        rng = np.random.default_rng((77, TIER_ORDER.index(tier), int(s * 100)))
        _, score, counts, kk = bootstrap(
            splits, budget_multiplier=budget, safety_ratio=s,
            risk_multiplier=np.array([1.0, *RISK[tier]]), high_cap_ratio=1.0,
            rng=rng, k=k, with_counts=True)
        if max(cp(int(c), kk) for c in counts) <= bound and (best is None or score > best[0]):
            best = (score, s)
    return best


def main() -> int:
    artifact = json.loads(_TEAM_ARTIFACT_JSON)
    base, data = load_splits(REPO_ROOT / "build/search-matrices.npz")
    budgets = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))
    weights = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}

    # uniform shrinkage first, then the per-model shape the measurements argue for
    trials = [("없음 (현재)", [0.0, 0.0, 0.0])]
    trials += [(f"균일 {x:.1f}", [x, x, x]) for x in (0.2, 0.4, 0.6, 0.8, 1.0)]
    trials += [
        ("ax31만 0.5", [0.0, 0.5, 0.0]),
        ("ax31만 1.0", [0.0, 1.0, 0.0]),
        ("ax31 1.0 / axk1 0.3", [0.0, 1.0, 0.3]),
        ("light+ax31 0.7", [0.7, 0.7, 0.0]),
    ]
    print(f"{'수축 (light/ax31/axk1)':26}" + "".join(f"{t:>12}" for t in TIER_ORDER) + f"{'가중합':>10}")
    for label, lam in trials:
        scores = predicted_scores(artifact, lam)
        splits = [(scores[i], *base[i][1:]) for i in range(2)]
        total, cells = 0.0, ""
        for tier in TIER_ORDER:
            got = best_at_bound(splits, tier, budgets[tier])
            if got is None:
                cells += f"{'실패':>12}"
                total = float("nan")
                continue
            cells += f"{got[0]:>12.4f}"
            total += weights[tier] * got[0]
        print(f"{label:26}{cells}{total:>10.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
