# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Fit the score heads with a binomial link instead of least squares.

The score is not a free real number: it is the fraction of generations that
came out correct, out of num_generations (2 for 86% of rows, 4 for the rest).
On Train it lands on 0.0 for 25.7% of rows and 1.0 for 64.9% -- 90.6% sits on
a boundary. Least squares is the wrong loss for that shape twice over:

  * the fit is unbounded, so the router clamps w.x into [0,1] at inference and
    throws away whatever the model was trying to say beyond the edge;
  * the loss keeps paying attention to rows it already has right, because
    pushing a fitted 1.4 down to 1.0 reduces squared error as much as fixing a
    genuinely wrong row.

A logit link fixes both without adding a single parameter -- same 266
coefficients, same dot product at inference, one sigmoid on top. That matters
here because the container ships stdlib only and because the thing we are most
afraid of is variance, not bias.

Fitted by IRLS with an L2 penalty, which is the GLM analogue of ridge, and
evaluated the only way that counts: allocator scores at a matched overrun
bound on both splits.

    py -3.13 training/experiment_glm.py
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
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    load_input,
    load_outcomes,
)
from search_standalone import SAFETY_GRID, TIER_ORDER, bootstrap, load_splits  # noqa: E402

RISK = {"fast": (1.0, 1.5), "balanced": (1.0, 2.5), "premium": (1.0, 6.0)}
WEIGHT = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
ALPHAS = (1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0)


def collect(split, hash_bins):
    batch = load_input(REPO_ROOT / f"data/materialized/{split}/inputs.json")
    index = {(o.episode_id, o.model_id): o
             for o in load_outcomes(REPO_ROOT / f"data/{split}/outcomes.json").outcomes}
    X, y, n_gen = [], [], []
    for e in batch.episodes:
        if (e.episode_id, MODEL_IDS[0]) not in index:
            continue
        X.append(_team_raw_feature_vector(e, hash_bins))
        y.append([float(index[(e.episode_id, m)].score) for m in MODEL_IDS])
        n_gen.append([float(index[(e.episode_id, m)].num_generations) for m in MODEL_IDS])
    return np.array(X, float), np.array(y, float), np.array(n_gen, float)


def fit_binomial(Z, y, weights, alpha, iters=25):
    """IRLS with an L2 penalty. Z carries an intercept column of ones.

    Weights are the generation counts: a row that scored 1/2 is two Bernoulli
    trials, not one observation, and the fit should know that.
    """
    beta = np.zeros(Z.shape[1])
    penalty = alpha * np.eye(Z.shape[1])
    penalty[0, 0] = 0.0  # never shrink the intercept
    for _ in range(iters):
        eta = np.clip(Z @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = weights * np.clip(mu * (1.0 - mu), 1e-6, None)
        z = eta + (y - mu) / np.clip(mu * (1.0 - mu), 1e-6, None)
        lhs = Z.T @ (w[:, None] * Z) + penalty
        rhs = Z.T @ (w * z)
        new = np.linalg.solve(lhs, rhs)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def main() -> int:
    artifact = json.loads(_TEAM_ARTIFACT_JSON)
    bins = artifact["hash_bins"]
    mean = np.array(artifact["feature_mean"])
    scale = np.array(artifact["feature_scale"])

    Xtr, ytr, gtr = collect("train", bins)
    Xdv, ydv, gdv = collect("dev", bins)
    Ztr = np.column_stack([np.ones(len(Xtr)), (Xtr - mean) / scale])
    Zdv = np.column_stack([np.ones(len(Xdv)), (Xdv - mean) / scale])

    # alpha by 5-fold out-of-fold deviance, per model, on Train only
    print("alpha 선택 (out-of-fold 이항 deviance, 5-fold):")
    chosen = {}
    folds = np.arange(len(Ztr)) % 5
    for j, m in enumerate(MODEL_IDS):
        best = None
        for a in ALPHAS:
            dev = 0.0
            for f in range(5):
                tr, te = folds != f, folds == f
                b = fit_binomial(Ztr[tr], ytr[tr, j], gtr[tr, j], a)
                p = np.clip(1.0 / (1.0 + np.exp(-np.clip(Ztr[te] @ b, -30, 30))), 1e-9, 1 - 1e-9)
                t = ytr[te, j]
                dev += -2.0 * np.sum(gtr[te, j] * (t * np.log(p) + (1 - t) * np.log(1 - p)))
            if best is None or dev < best[0]:
                best = (dev, a)
        chosen[m] = best[1]
        print(f"  {m:12} alpha={best[1]:>7g}  deviance={best[0]:.1f}")

    pred = {}
    for tag, Z in (("dev", Zdv), ("train", Ztr)):
        cols = []
        for j, m in enumerate(MODEL_IDS):
            b = fit_binomial(Ztr, ytr[:, j], gtr[:, j], chosen[m])
            cols.append(1.0 / (1.0 + np.exp(-np.clip(Z @ b, -30, 30))))
        pred[tag] = np.column_stack(cols)

    base, data = load_splits(REPO_ROOT / "build/search-matrices.npz")
    budgets = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))
    print(f"\n{'':22}" + "".join(f"{t:>12}" for t in TIER_ORDER) + f"{'가중합':>10}")
    for label, scores in (("현재 (최소제곱)", [base[0][0], base[1][0]]),
                          ("GLM (로짓 링크)", [pred["dev"], pred["train"]])):
        splits = [(scores[i], *base[i][1:]) for i in range(2)]
        total, cells = 0.0, ""
        for tier in TIER_ORDER:
            best = None
            for s in SAFETY_GRID:
                rng = np.random.default_rng((11, TIER_ORDER.index(tier), int(s * 100)))
                _, score, counts, k = bootstrap(
                    splits, budget_multiplier=budgets[tier], safety_ratio=s,
                    risk_multiplier=np.array([1.0, *RISK[tier]]), high_cap_ratio=1.0,
                    rng=rng, k=2000, with_counts=True)
                if max(cp(int(c), k) for c in counts) <= 0.005 and (best is None or score > best):
                    best = score
            cells += f"{best:>12.4f}"
            total += WEIGHT[tier] * best
        print(f"{label:22}{cells}{total:>10.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
