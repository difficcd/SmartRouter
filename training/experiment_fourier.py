# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Nonlinearity without leaving ridge: trigonometric basis expansion.

==님's idea, and it lands on a known result -- expanding a feature into
sin/cos pairs at several frequencies lets a LINEAR model approximate an RBF
kernel (random Fourier features). The router keeps its closed-form ridge fit,
its heavy regularization, and a single dot product at inference, so nothing
about the container or the calibration machinery changes.

It also settles the GBM question cheaply. The real issue with GBM is not
overfitting in the abstract -- it is that our evaluation fits on Train and
bootstraps on Dev+Train, so a flexible model's score comes back inflated
exactly where it needs the most scrutiny. This runs at ridge's variance
profile instead, which means the number it produces is trustworthy:

  * if the expansion helps, nonlinear structure exists in these features and
    GBM becomes worth the day it would cost;
  * if it does nothing, the structure is not there and GBM would not find it
    either.

Expansion is confined to the 10 hand-built dense features. The 256 hash slots
are excluded on purpose: with thousands of tokens colliding into 256 bins,
each slot is already a mixture, and expanding a mixture buys noise. 10 x 3
frequencies x 2 = 60 extra columns against 1760 Train rows.

    py -3.13 training/experiment_fourier.py
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
    _TEAM_DENSE_FEATURE_NAMES,
    _team_raw_feature_vector,
)
from ossp_router.protocol import MODEL_IDS, load_input, load_outcomes  # noqa: E402
from search_standalone import SAFETY_GRID, TIER_ORDER, bootstrap, load_splits  # noqa: E402
from train_router import fit_ridge, oof_predictions, predict_ridge  # noqa: E402

RISK = {"fast": (1.0, 1.5), "balanced": (1.0, 2.5), "premium": (1.0, 6.0)}
WEIGHT = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
ALPHAS = (1.0, 10.0, 100.0, 300.0, 1000.0, 10000.0)
DENSE = len(_TEAM_DENSE_FEATURE_NAMES)


def collect(split, bins):
    batch = load_input(REPO_ROOT / f"data/materialized/{split}/inputs.json")
    index = {(o.episode_id, o.model_id): o
             for o in load_outcomes(REPO_ROOT / f"data/{split}/outcomes.json").outcomes}
    X, y = [], []
    for e in batch.episodes:
        if (e.episode_id, MODEL_IDS[0]) not in index:
            continue
        X.append(_team_raw_feature_vector(e, bins))
        y.append([float(index[(e.episode_id, m)].score) for m in MODEL_IDS])
    return np.array(X, float), np.array(y, float)


def expand(X, mean, scale, freqs):
    """Original columns, plus sin/cos of the standardized dense block."""
    if not freqs:
        return X
    z = (X[:, :DENSE] - mean[:DENSE]) / scale[:DENSE]
    z = np.clip(z, -4.0, 4.0)   # the tail is heavy; keep the basis in range
    extra = [f(w * z) for w in freqs for f in (np.sin, np.cos)]
    return np.hstack([X] + extra)


def main() -> int:
    artifact = json.loads(_TEAM_ARTIFACT_JSON)
    bins = artifact["hash_bins"]
    mean = np.array(artifact["feature_mean"])
    scale = np.array(artifact["feature_scale"])
    Xtr, ytr = collect("train", bins)
    Xdv, _ = collect("dev", bins)

    base, data = load_splits(REPO_ROOT / "build/search-matrices.npz")
    budgets = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))

    print(f"{'주파수':22}{'열 수':>7}{'OOF MSE':>10}" +
          "".join(f"{t:>11}" for t in TIER_ORDER) + f"{'가중합':>10}")
    for label, freqs in (("없음 (현재)", ()), ("0.5, 1, 2", (0.5, 1.0, 2.0)),
                         ("1, 2, 4", (1.0, 2.0, 4.0)), ("0.25, 0.5, 1", (0.25, 0.5, 1.0))):
        Etr = expand(Xtr, mean, scale, freqs)
        Edv = expand(Xdv, mean, scale, freqs)
        best = min(ALPHAS, key=lambda a: float(
            ((oof_predictions(Etr, ytr, folds=5, alpha=a) - ytr) ** 2).mean()))
        mse = float(((oof_predictions(Etr, ytr, folds=5, alpha=best) - ytr) ** 2).mean())
        m2, s2, inter, coef = fit_ridge(Etr, ytr, best)
        preds = [np.clip(predict_ridge(E, m2, s2, inter, coef), 0.0, 1.0) for E in (Edv, Etr)]
        splits = [(preds[i], *base[i][1:]) for i in range(2)]
        total, cells = 0.0, ""
        for tier in TIER_ORDER:
            top = None
            for s in SAFETY_GRID:
                rng = np.random.default_rng((71, TIER_ORDER.index(tier), int(s * 100)))
                _, score, counts, k = bootstrap(
                    splits, budget_multiplier=budgets[tier], safety_ratio=s,
                    risk_multiplier=np.array([1.0, *RISK[tier]]), high_cap_ratio=1.0,
                    rng=rng, k=1500, with_counts=True)
                if max(cp(int(c), k) for c in counts) <= 0.005 and (top is None or score > top):
                    top = score
            cells += f"{top:>11.4f}"
            total += WEIGHT[tier] * top
        print(f"{label:22}{Etr.shape[1]:>7}{mse:>10.5f}{cells}{total:>10.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
