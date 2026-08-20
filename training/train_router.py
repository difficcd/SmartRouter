# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Fit the six ridge heads (score x3, log-cost x3) on public Train.

Offline only -- NumPy allowed here even though the shipped inference path
(the "difficcd team router additions" section of src/ossp_router/heuristic.py)
is pure standard library. Produces training/artifact.json with placeholder
safety ratios (1.0) and risk multipliers (1.0); run calibrate_safety.py
afterward to set those from real held-out behavior, then bake_artifact.py
to embed the result into heuristic.py before building the container.

Imports the feature function directly from heuristic.py (rather than keeping
a second copy here) so training and inference can never silently diverge.

    py -3.13 training/train_router.py \
        --train-input data/materialized/train/inputs.json \
        --train-outcomes data/train/outcomes.json \
        --hash-bins 256 \
        --out training/artifact.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BAGGING_ROUNDS = 30

from ossp_router.heuristic import (  # noqa: E402
    _TEAM_DENSE_FEATURE_NAMES,
    _team_raw_feature_vector,
)
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    InputBatch,
    OutcomeBatch,
    RoutingPolicy,
    load_bundled_policy,
    load_input,
    load_outcomes,
    policy_sha256,
)


def _outcome_cost(outcome, policy: RoutingPolicy) -> float:
    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    cost = (
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )
    value = float(cost)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("학습 outcome의 모델 비용은 0보다 커야 합니다.")
    return value


def build_training_matrix(
    inputs: InputBatch, outcomes: OutcomeBatch, policy: RoutingPolicy, hash_bins: int
) -> Tuple[Any, Any]:
    outcome_index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    expected = {
        (episode.episode_id, model_id)
        for episode in inputs.episodes
        for model_id in MODEL_IDS
    }
    if set(outcome_index) != expected:
        raise ValueError("Train outcome 행렬이 입력과 모델 전체를 포함하지 않습니다.")

    matrix = np.asarray(
        [_team_raw_feature_vector(episode, hash_bins) for episode in inputs.episodes],
        dtype=np.float64,
    )
    targets = []
    for episode in inputs.episodes:
        rows = [outcome_index[(episode.episode_id, m)] for m in MODEL_IDS]
        scores = [float(row.score) for row in rows]
        log_costs = [math.log(_outcome_cost(row, policy)) for row in rows]
        targets.append(scores + log_costs)
    return matrix, np.asarray(targets, dtype=np.float64)


def _ridge_solve(standardized, targets, alpha: float):
    """Assumes `standardized` is already (X-mean)/scale. Split out of fit_ridge
    so bagging can reuse ONE fixed standardization across bootstrap resamples
    of the rows (see bag_ridge) instead of each resample computing its own
    mean/scale, which would make the resulting coefficients incomparable."""
    intercept = targets.mean(axis=0)
    centered = targets - intercept

    rows, columns = standardized.shape
    if rows <= columns:
        system = standardized @ standardized.T + alpha * np.eye(rows)
        coefficients = standardized.T @ np.linalg.solve(system, centered)
    else:
        system = standardized.T @ standardized + alpha * np.eye(columns)
        coefficients = np.linalg.solve(system, standardized.T @ centered)
    return intercept, coefficients


def fit_ridge(matrix, targets, alpha: float):
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    intercept, coefficients = _ridge_solve(standardized, targets, alpha)
    return mean, scale, intercept, coefficients


def bag_ridge(matrix, targets, alpha: float, *, rounds: int, rng: np.random.Generator):
    """Bootstrap-aggregate ridge: refit on `rounds` bootstrap resamples of the
    rows (with replacement) and average the coefficients. Because the model is
    linear, averaging B sets of coefficients is exactly equivalent to averaging
    B models' predictions (mean(w_b @ x) == mean(w_b) @ x) -- so this adds zero
    runtime cost to the shipped predictor, only to this offline fit. Uses ONE
    fixed mean/scale from the full (non-resampled) data so every resample's
    coefficients are in the same units and can be validly averaged."""
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale

    rows = matrix.shape[0]
    intercept_sum = None
    coefficients_sum = None
    for _ in range(rounds):
        index = rng.integers(0, rows, size=rows)
        intercept, coefficients = _ridge_solve(standardized[index], targets[index], alpha)
        if intercept_sum is None:
            intercept_sum, coefficients_sum = intercept, coefficients
        else:
            intercept_sum = intercept_sum + intercept
            coefficients_sum = coefficients_sum + coefficients
    return mean, scale, intercept_sum / rounds, coefficients_sum / rounds


def predict_ridge(matrix, mean, scale, intercept, coefficients):
    return (matrix - mean) / scale @ coefficients + intercept


def oof_predictions(matrix, targets, *, folds: int, alpha: float):
    rows = matrix.shape[0]
    predictions = np.empty_like(targets)
    fold_ids = np.arange(rows) % folds
    for fold in range(folds):
        validation = fold_ids == fold
        training = ~validation
        mean, scale, intercept, coefficients = fit_ridge(matrix[training], targets[training], alpha)
        predictions[validation] = predict_ridge(matrix[validation], mean, scale, intercept, coefficients)
    return predictions


def select_alpha_single(matrix, targets, *, folds: int, candidates: Sequence[float], label: str) -> float:
    """targets here is ONE block (score-only or log-cost-only) -- fit and pick
    alpha for that block alone. A shared alpha across score+cost let score's
    much larger MSE dominate the objective and left the cost heads
    under-regularized."""
    best_alpha = candidates[0]
    best_mse = math.inf
    for alpha in candidates:
        predictions = oof_predictions(matrix, targets, folds=folds, alpha=alpha)
        mse = float(np.mean((predictions - targets) ** 2))
        print(f"  [{label}] alpha={alpha:>10.4g}  mse={mse:.5f}")
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    return best_alpha


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-outcomes", type=Path, required=True)
    parser.add_argument("--hash-bins", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    policy = load_bundled_policy()
    inputs = load_input(args.train_input)
    outcomes = load_outcomes(args.train_outcomes)

    print(f"학습 문항 수: {len(inputs.episodes)}  hash_bins={args.hash_bins}")
    matrix, targets = build_training_matrix(inputs, outcomes, policy, args.hash_bins)
    print(f"특징 행렬: {matrix.shape}  타깃: {targets.shape}")

    model_count = len(MODEL_IDS)
    score_targets = targets[:, :model_count]
    cost_targets = targets[:, model_count:]

    candidates = [10.0**exponent for exponent in range(-1, 7)]
    print("alpha 후보 탐색 -- score와 cost를 따로 고른다 (out-of-fold, 5-fold):")
    score_alpha = select_alpha_single(matrix, score_targets, folds=5, candidates=candidates, label="score")
    cost_alpha = select_alpha_single(matrix, cost_targets, folds=5, candidates=candidates, label="cost ")
    print(f"선택된 alpha: score={score_alpha:g}  cost={cost_alpha:g}")

    # exp()로 log-cost를 원래 스케일로 되돌리면 Jensen 부등식 때문에 구조적으로
    # 과소추정된다(E[exp(residual)] > exp(E[residual]) = 1, residual이 정확히
    # 0으로 안 맞아떨어지는 한). Duan(1983) smearing estimator: 재학습 없이
    # out-of-fold 잔차의 exp() 평균을 모델별로 곱해서 이 편향을 보정한다.
    cost_oof = oof_predictions(matrix, cost_targets, folds=5, alpha=cost_alpha)
    cost_smear = np.exp(cost_targets - cost_oof).mean(axis=0)
    print(
        "cost smearing 보정계수(모델별, out-of-fold): "
        + "  ".join(f"{m}={s:.4f}" for m, s in zip(MODEL_IDS, cost_smear))
    )

    # Bagging: refit on BAGGING_ROUNDS bootstrap resamples of the 1,760 rows and
    # average the coefficients (exactly equivalent to averaging predictions,
    # since the model is linear -- see bag_ridge's docstring). Aimed at the
    # tail-error problem that's been driving the fast-tier safety margin: a
    # single ridge fit can happen to be thrown off by whichever rows landed in
    # it, bagging smooths that out. Zero added cost at inference time.
    bag_rng = np.random.default_rng(20260820)
    score_mean, score_scale, score_intercept, score_coefficients = bag_ridge(
        matrix, score_targets, score_alpha, rounds=BAGGING_ROUNDS, rng=bag_rng
    )
    cost_mean, cost_scale, cost_intercept, cost_coefficients = bag_ridge(
        matrix, cost_targets, cost_alpha, rounds=BAGGING_ROUNDS, rng=bag_rng
    )
    assert np.allclose(score_mean, cost_mean) and np.allclose(score_scale, cost_scale)
    mean, scale = score_mean, score_scale

    def head_dict(intercept: float, coefficients) -> dict:
        return {"intercept": float(intercept), "coefficients": [float(c) for c in coefficients]}

    artifact = {
        "artifact_type": "team-router-v1",
        "schema_version": 1,
        "feature_version": "team-features-v1",
        "hash_algorithm": "fnv1a64-signed-word-1-2",
        "hash_bins": args.hash_bins,
        "dense_feature_names": list(_TEAM_DENSE_FEATURE_NAMES),
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "feature_mean": [float(v) for v in mean],
        "feature_scale": [float(v) for v in scale],
        "score_heads": {
            model_id: head_dict(score_intercept[i], score_coefficients[:, i])
            for i, model_id in enumerate(MODEL_IDS)
        },
        "log_cost_heads": {
            model_id: head_dict(cost_intercept[i], cost_coefficients[:, i])
            for i, model_id in enumerate(MODEL_IDS)
        },
        "cost_smear": {
            model_id: float(cost_smear[i]) for i, model_id in enumerate(MODEL_IDS)
        },
        "tier_safety_ratios": {tier: 1.0 for tier in TIERS},  # placeholder -- calibrate next
        # placeholder -- per-tier now, since fast/premium want opposite values
        "risk_multiplier": {
            tier: {model_id: 1.0 for model_id in MODEL_IDS} for tier in TIERS
        },
        "high_cap_ratio": {tier: 1.0 for tier in TIERS},  # placeholder -- calibrate next
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"OK: {args.out}에 학습 파일을 저장했습니다.")
    print("주의: tier_safety_ratios/risk_multiplier는 아직 자리표시자(1.0)입니다.")
    print("      training/calibrate_safety.py -> training/bake_artifact.py 순으로 이어서 실행할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
