# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Fit the budget-side cost head to an upper quantile instead of the mean.

==님's idea, and it is the same target IDEA item 9 ("asymmetric loss for cost")
was aiming at. The motivation is not accuracy, which is where I mis-priced it
the first time. The oracle decomposition measured "what if the MEAN cost were
predicted perfectly" and answered +0.0046, so I deprioritised the whole cost
family. That answered the wrong question.

What the router actually needs from a cost estimate is not to be centred; it
is to not be exceeded. A mean fit is under the truth half the time, and every
one of those halves eats into the same margin the safety ratio is paying for.
A q-th quantile fit is deliberately above the truth (1-q) of the time, so the
margin is built into the prediction rather than bought with a tighter cap.

Why this matters here specifically: premium spends only 65.4% of its budget,
and the 34.6% left over cannot be taken by raising safety -- 0.60 -> 0.70 moves
the overrun bound from 0.16% to 4.64%, a 29x jump, and that point is already
the best of 48 risk combinations. The frontier has to move, not the operating
point on it.

Fitted by IRLS on the pinball loss with an L2 penalty, so it stays a closed-form
family: same 326 coefficients, same dot product at inference, no new container
dependency. Only the BUDGET head changes; the ranking head keeps its median-ish
fit, because ranking wants "which is cheaper", not "what is the ceiling".

    py -3.13 training/experiment_quantile_cost.py
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
ALPHAS = (10.0, 100.0, 1000.0, 10000.0)


def collect(split, artifact):
    batch = load_input(REPO_ROOT / f"data/materialized/{split}/inputs.json")
    index = {(o.episode_id, o.model_id): o
             for o in load_outcomes(REPO_ROOT / f"data/{split}/outcomes.json").outcomes}
    policy = load_bundled_policy()
    unit = float(policy.token_unit)
    rate_in = {m: float(policy.models[m].input_token_rate) / unit for m in MODEL_IDS}
    rate_out = {m: float(policy.models[m].output_token_rate) / unit for m in MODEL_IDS}
    bmean = np.array(artifact["basis_mean"])
    bscale = np.array(artifact["basis_scale"])
    X, cost = [], []
    for episode in batch.episodes:
        if (episode.episode_id, MODEL_IDS[0]) not in index:
            continue
        X.append(_team_expand_basis(
            _team_raw_feature_vector(episode, artifact["hash_bins"]), bmean, bscale))
        row = []
        for m in MODEL_IDS:
            o = index[(episode.episode_id, m)]
            row.append(o.input_tokens * rate_in[m] + o.output_tokens * rate_out[m])
        cost.append(row)
    return np.array(X, dtype=float), np.array(cost, dtype=float)


def fit_quantile(Z, y, q, alpha, iterations=40, floor=1e-4):
    """Pinball-loss regression by iteratively reweighted least squares.

    Pinball loss is |residual| weighted q above the line and (1-q) below, and
    an L1-shaped loss becomes a weighted L2 problem with weights 1/|residual|.
    Iterating that converges to the quantile fit while every step stays a
    closed-form solve -- which is what keeps this shippable.

    Column 0 of Z is the intercept and it is NEVER penalised. fit_ridge in
    train_router.py gets this right by centring the targets; the first run of
    this experiment put a ones column into a penalised design instead, and in
    LOG space shrinking the intercept toward zero drags predictions toward
    exp(0)=1. Costs live around 1e-4..1e-2, so that over-predicted dev batch
    totals by 84x/41x/3.7x and left the MEDIAN fit under-predicting only 1.3%
    of the time instead of the ~50% a median must give. Leaving the intercept
    free also matters on its own terms: the intercept rising with q is exactly
    the mechanism this experiment exists to measure.
    """
    penalty = alpha * np.eye(Z.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y)
    for _ in range(iterations):
        residual = y - Z @ beta
        weight = np.where(residual >= 0, q, 1.0 - q) / np.maximum(np.abs(residual), floor)
        lhs = Z.T @ (weight[:, None] * Z) + penalty
        new = np.linalg.solve(lhs, Z.T @ (weight * y))
        if np.max(np.abs(new - beta)) < 1e-9:
            return new
        beta = new
    return beta


def monotone(pred):
    """The clamp heuristic.py applies before the allocator ever sees a cost.

    light <= ax31 <= axk1 is forced there, so any arm compared against the
    shipped router has to be clamped the same way -- independent per-model
    quantile fits can cross, and an unclamped arm would be measured under a
    rule the shipped one does not play by.
    """
    out = pred.copy()
    out[:, 1] = np.maximum(out[:, 1], out[:, 0] * (1.0 + 1e-12))
    out[:, 2] = np.maximum(out[:, 2], out[:, 1] * (1.0 + 1e-12))
    return out


def shipped_budget_cost(X, artifact):
    """The budget cost head main actually ships, used as the control arm.

    Read straight out of the baked artifact rather than refitted, so the
    control cannot drift from v20b through some detail of the refit. This is
    a log-space ridge times a per-model Duan smearing factor (1.32 / 1.48 /
    1.73), which already inflates predictions -- the quantile arm is not
    adding inflation that was absent, it is replacing a constant multiplier
    with a feature-dependent one.
    """
    mean = np.array(artifact["feature_mean"])
    scale = np.array(artifact["feature_scale"])
    Zs = (X - mean) / scale
    cols = []
    for m in MODEL_IDS:
        head = artifact["log_cost_heads"][m]
        raw = Zs @ np.array(head["coefficients"]) + float(head["intercept"])
        cols.append(np.exp(np.clip(raw, -50, 50)) * float(artifact["cost_smear"][m]))
    return monotone(np.column_stack(cols))


def main() -> int:
    artifact = json.loads(_TEAM_ARTIFACT_JSON)
    Xtr, Ctr = collect("train", artifact)
    Xdv, Cdv = collect("dev", artifact)
    mean, scale = Xtr.mean(axis=0), Xtr.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    Ztr = np.column_stack([np.ones(len(Xtr)), (Xtr - mean) / scale])
    Zdv = np.column_stack([np.ones(len(Xdv)), (Xdv - mean) / scale])
    log_tr = np.log(np.maximum(Ctr, 1e-9))

    base, data = load_splits(REPO_ROOT / "build/v20-matrices.npz")
    budgets = dict(zip(TIER_ORDER, data["budget_multipliers"].tolist()))
    # each tier's shipped risk multipliers, so the comparison is against v20b
    # rather than against one arbitrary setting reused for all three
    params = json.loads(
        (REPO_ROOT / "build/v20-target-0.003.json").read_text(encoding="utf-8"))

    print("예산용 cost head를 상위 분위로 적합. 랭킹용은 그대로.\n")
    print(f"{'구성':10}{'dev 배치합 예측/실제':>22}{'과소예측 비율':>14}"
          + "".join(f"{t:>11}" for t in TIER_ORDER) + f"{'가중합':>10}{'최악상한':>10}")
    # alpha is part of the arm, not a constant. The shipped control had its
    # cost alpha chosen by out-of-fold log MSE, and that optimum measured 300
    # -- so running every quantile arm at 1000 would compare a tuned control
    # against over-regularised challengers.
    for label, q, alpha in (("v20b", None, 0.0),
                            ("0.65 a300", 0.65, 300.0),
                            ("0.75 a300", 0.75, 300.0),
                            ("0.75 a100", 0.75, 100.0)):
        if q is None:
            pred = {"train": shipped_budget_cost(Xtr, artifact),
                    "dev": shipped_budget_cost(Xdv, artifact)}
        else:
            betas = [fit_quantile(Ztr, log_tr[:, j], q, alpha)
                     for j in range(len(MODEL_IDS))]
            pred = {
                tag: monotone(np.column_stack(
                    [np.exp(np.clip(Z @ b, -50, 50)) for b in betas]))
                for tag, Z in (("train", Ztr), ("dev", Zdv))
            }
        # NO batch-sum correction here, and that is the whole point. The
        # existing pipeline rescales exp(log-fit) so the predicted batch total
        # matches the real one, which is right for a mean fit fighting Jensen.
        # Applied to a quantile fit it cancels exactly the upward bias we are
        # buying -- the first run of this experiment did that and the
        # under-prediction rate refused to move (24.7% -> 24.0% from q=0.50 to
        # 0.65). The quantile is supposed to sit above the truth; the safety
        # search then compensates by choosing a looser safety_ratio, and the
        # net of those two is what we are here to measure.
        ratio = pred["dev"].sum(axis=0) / Cdv.sum(axis=0)
        under = float((pred["dev"] < Cdv).mean())

        splits = [(base[i][0], pred[tag], base[i][2], base[i][3], base[i][4])
                  for i, tag in enumerate(("dev", "train"))]
        total, cells, worst = 0.0, "", 0.0
        for tier in TIER_ORDER:
            best = None
            shipped = params[tier]
            risk = np.array([1.0, shipped["risk_mid"], shipped["risk_high"]])
            for s in SAFETY_GRID:
                rng = np.random.default_rng((17, TIER_ORDER.index(tier), int(s * 100)))
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
        levels = " ".join(f"{v:.2f}" for v in ratio)
        print(f"{label:>10}{levels:>22}{under:>13.1%}{cells}{total:>10.4f}{worst:>9.2%}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
