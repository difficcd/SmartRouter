# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap-aggregated ridge coefficients, re-tried under a fixed statistic.

v4 rejected bagging and the rejection is confounded. Its own commit message
records that premium's loss came with "risk_high re-tuned to the grid ceiling
of 4.0" -- and v6, the very next version, widened those grids and called the
widening a real improvement. A search pinned against a boundary cannot price a
change fairly. v4 also predates v16, which found the safety search was judged
by a statistic that could not resolve its own target.

The mechanism is variance reduction in the coefficients, and v20 has since
widened the design from 266 to 326 columns, so there is more variance to
reduce now than at v3. On a true holdout (fit Train, predict Dev) bagging
improves both heads -- score MSE 0.16035 -> 0.15853, log-cost 0.55672 ->
0.55468 -- and all ten seeds beat the single fit on both.

Three arms, because two would confound bagging with refitting:

    shipped   the baked v20b artifact -- anchors the harness
    single    my own refit, no bagging -- the matched control
    bagged    my own refit, 300 bags   -- the treatment

single vs bagged is the experiment. shipped only reports whether my refit
reproduces what main ships.

    py -3.13 training/experiment_bagging.py
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
    _team_expand_basis,
    _team_raw_feature_vector,
)
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)
from search_standalone import SAFETY_GRID, TIER_ORDER, bootstrap, load_splits  # noqa: E402

WEIGHT = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
SCORE_ALPHA = 1000.0   # both picked by Train out-of-fold MSE, and the optimum
COST_ALPHA = 300.0     # is the same for the single and the bagged arm
BAGS = 300             # where seed spread stops mattering: sigma 0.00006/0.00036


def collect(split, artifact):
    batch = load_input(REPO_ROOT / f"data/materialized/{split}/inputs.json")
    index = {(o.episode_id, o.model_id): o
             for o in load_outcomes(REPO_ROOT / f"data/{split}/outcomes.json").outcomes}
    policy = load_bundled_policy()
    unit = float(policy.token_unit)
    rin = {m: float(policy.models[m].input_token_rate) / unit for m in MODEL_IDS}
    rout = {m: float(policy.models[m].output_token_rate) / unit for m in MODEL_IDS}
    bmean = np.array(artifact["basis_mean"])
    bscale = np.array(artifact["basis_scale"])
    X, S, C = [], [], []
    for ep in batch.episodes:
        if (ep.episode_id, MODEL_IDS[0]) not in index:
            continue
        X.append(_team_expand_basis(
            _team_raw_feature_vector(ep, artifact["hash_bins"]), bmean, bscale))
        S.append([float(index[(ep.episode_id, m)].score) for m in MODEL_IDS])
        # fixed_cost is 0 for all three models today, but the official scorer
        # includes it and so do train_router and calibrate_safety. Leaving it
        # out here would make this script quietly disagree with them the day a
        # policy sets it.
        C.append([float(policy.models[m].fixed_cost)
                  + index[(ep.episode_id, m)].input_tokens * rin[m]
                  + index[(ep.episode_id, m)].output_tokens * rout[m]
                  for m in MODEL_IDS])
    return np.array(X, float), np.array(S, float), np.array(C, float)


def fit(Z, y, alpha, bags=0, seed=0):
    """Ridge, optionally averaged over `bags` row resamples.

    Standardisation is computed once on the full block rather than per bag, so
    the averaged coefficients stay ONE usable coefficient set -- bagging has to
    remain free at inference or it cannot ship inside the container.
    """
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Zs = (Z - mu) / sd
    level = y.mean(0)
    centred = y - level
    eye = alpha * np.eye(Zs.shape[1])
    if bags == 0:
        return mu, sd, level, np.linalg.solve(Zs.T @ Zs + eye, Zs.T @ centred)
    rng = np.random.default_rng(seed)
    acc = np.zeros((Zs.shape[1], y.shape[1]))
    for _ in range(bags):
        pick = rng.integers(0, len(Zs), len(Zs))
        Zb, yb = Zs[pick], centred[pick]
        acc += np.linalg.solve(Zb.T @ Zb + eye, Zb.T @ yb)
    return mu, sd, level, acc / bags


def apply(Z, f):
    mu, sd, level, co = f
    return (Z - mu) / sd @ co + level


def oof(Z, y, alpha, bags, folds=5):
    out = np.empty_like(y)
    for k in range(folds):
        va = np.arange(len(Z)) % folds == k
        out[va] = apply(Z[va], fit(Z[~va], y[~va], alpha, bags, seed=1000 + k))
    return out


def monotone(pred):
    """light <= ax31 <= axk1, the clamp heuristic.py applies before allocating."""
    out = pred.copy()
    out[:, 1] = np.maximum(out[:, 1], out[:, 0] * (1.0 + 1e-12))
    out[:, 2] = np.maximum(out[:, 2], out[:, 1] * (1.0 + 1e-12))
    return out


def from_artifact(X, artifact, block, smear_key=None):
    mean = np.array(artifact["feature_mean"])
    scale = np.array(artifact["feature_scale"])
    Zs = (X - mean) / scale
    cols = []
    for m in MODEL_IDS:
        head = artifact[block][m]
        raw = Zs @ np.array(head["coefficients"]) + float(head["intercept"])
        if smear_key is None:
            cols.append(np.clip(raw, 0.0, 1.0))
        else:
            cols.append(np.exp(np.clip(raw, -50, 50)) * float(artifact[smear_key][m]))
    out = np.column_stack(cols)
    return out if smear_key is None else monotone(out)


def main() -> int:
    artifact = json.loads(_TEAM_ARTIFACT_JSON)
    Xtr, Str, Ctr = collect("train", artifact)
    Xdv, _, _ = collect("dev", artifact)
    log_tr = np.log(np.maximum(Ctr, 1e-9))

    base, data = load_splits(REPO_ROOT / "build/v20-matrices.npz")
    budgets = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))
    params = json.loads(
        (REPO_ROOT / "build/v20-target-0.003.json").read_text(encoding="utf-8"))

    def refit(bags):
        score_head = fit(Xtr, Str, SCORE_ALPHA, bags, seed=7)
        cost_head = fit(Xtr, log_tr, COST_ALPHA, bags, seed=7)
        # Duan smearing exactly as train_router.py's classic path builds it:
        # the per-episode mean of exp(residual), measured out-of-fold.
        smear = np.exp(log_tr - oof(Xtr, log_tr, COST_ALPHA, bags)).mean(axis=0)
        return {
            tag: (np.clip(apply(Z, score_head), 0.0, 1.0),
                  monotone(np.exp(np.clip(apply(Z, cost_head), -50, 50)) * smear))
            for tag, Z in (("train", Xtr), ("dev", Xdv))
        }

    arms = {
        "shipped": {tag: (from_artifact(Z, artifact, "score_heads"),
                          from_artifact(Z, artifact, "log_cost_heads", "cost_smear"))
                    for tag, Z in (("train", Xtr), ("dev", Xdv))},
        "single": refit(0),
        f"bagged x{BAGS}": refit(BAGS),
    }

    print(f"배깅 재검토 -- score alpha={SCORE_ALPHA:g}, cost alpha={COST_ALPHA:g}\n")
    header = f"{'구성':>14}" + "".join(f"{t:>11}" for t in TIER_ORDER)
    print(header + f"{'가중합':>10}{'최악상한':>10}")
    for label, pred in arms.items():
        # split tuple is (pred_scores, pred_costs, pred_rank_costs, real_scores,
        # real_costs); classic ships one cost head for both roles, so the budget
        # prediction is reused as the ranking prediction.
        splits = [(pred[tag][0], pred[tag][1], pred[tag][1], base[i][3], base[i][4])
                  for i, tag in enumerate(("dev", "train"))]
        total, cells, worst = 0.0, "", 0.0
        for tier in TIER_ORDER:
            tuned = params[tier]
            risk = np.array([1.0, tuned["risk_mid"], tuned["risk_high"]])
            best = None
            for s in SAFETY_GRID:
                rng = np.random.default_rng((23, TIER_ORDER.index(tier), int(s * 100)))
                _, score, counts, k = bootstrap(
                    splits, budget_multiplier=budgets[tier], safety_ratio=s,
                    risk_multiplier=risk, high_cap_ratio=1.0,
                    rng=rng, k=1200, with_counts=True)
                bound = max(cp(int(c), k) for c in counts)
                if bound <= 0.005 and (best is None or score > best[0]):
                    best = (score, bound)
            if best is None:
                cells += f"{'실패':>11}"
                continue
            cells += f"{best[0]:>11.4f}"
            total += WEIGHT[tier] * best[0]
            worst = max(worst, best[1])
        print(f"{label:>14}{cells}{total:>10.4f}{worst:>9.2%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
