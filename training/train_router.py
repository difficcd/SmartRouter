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


def fit_ridge(matrix, targets, alpha):
    """alpha: a scalar (uniform ridge, original behavior) or a (columns,)
    array (per-feature regularization strength -- see _alpha_vector). The
    dual-form (rows <= columns) branch doesn't have a simple per-feature
    generalization and this dataset (1,760 rows, 266 features) always takes
    the primal branch below, so that branch stays scalar-only."""
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    intercept = targets.mean(axis=0)
    centered = targets - intercept

    rows, columns = standardized.shape
    if rows <= columns:
        system = standardized @ standardized.T + float(alpha) * np.eye(rows)
        coefficients = standardized.T @ np.linalg.solve(system, centered)
    else:
        alpha_vec = np.broadcast_to(alpha, (columns,))
        system = standardized.T @ standardized + np.diag(alpha_vec)
        coefficients = np.linalg.solve(system, standardized.T @ centered)
    return mean, scale, intercept, coefficients


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


N_DENSE_FEATURES = 10  # len(_TEAM_DENSE_FEATURE_NAMES); rest are hash slots
HASH_ALPHA_MULTIPLIER_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]


def _alpha_vector(alpha: float, hash_multiplier: float, columns: int):
    """Per-feature regularization: the 10 hand-picked dense features (length,
    hangul_ratio, ...) keep `alpha`; the remaining (hash-bin) columns get
    `alpha * hash_multiplier`. The dense features are direct, low-noise
    signals; the hash slots are word-hash collisions and much noisier, so
    letting them take a different (typically stronger) regularization than
    the dense block is a real degree of freedom the original single-alpha
    ridge didn't have."""
    vec = np.full(columns, alpha, dtype=np.float64)
    vec[N_DENSE_FEATURES:] *= hash_multiplier
    return vec


def select_hash_multiplier(
    matrix, targets, *, folds: int, alpha: float, candidates: Sequence[float], label: str
) -> float:
    best_multiplier = candidates[0]
    best_mse = math.inf
    columns = matrix.shape[1]
    for multiplier in candidates:
        alpha_vec = _alpha_vector(alpha, multiplier, columns)
        predictions = oof_predictions(matrix, targets, folds=folds, alpha=alpha_vec)
        mse = float(np.mean((predictions - targets) ** 2))
        print(f"  [{label}] hash_alpha_multiplier={multiplier:>6.2g}  mse={mse:.5f}")
        if mse < best_mse:
            best_mse = mse
            best_multiplier = multiplier
    return best_multiplier


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

    # The 10 hand-picked dense features (length, hangul_ratio, ...) are a
    # direct, low-noise signal; the 256 hash-bin slots are word-hash
    # collisions and much noisier. A single shared alpha regularizes both
    # equally -- search a separate multiplier for just the hash block.
    hash_candidates = HASH_ALPHA_MULTIPLIER_GRID
    print("해시 슬롯 정규화 배수 탐색 (손수 특징 10개는 alpha 그대로, 해시 256칸만 추가 배율):")
    score_hash_mult = select_hash_multiplier(
        matrix, score_targets, folds=5, alpha=score_alpha, candidates=hash_candidates, label="score"
    )
    cost_hash_mult = select_hash_multiplier(
        matrix, cost_targets, folds=5, alpha=cost_alpha, candidates=hash_candidates, label="cost "
    )
    print(f"선택된 해시 배수: score={score_hash_mult:g}  cost={cost_hash_mult:g}")
    columns = matrix.shape[1]
    score_alpha_vec = _alpha_vector(score_alpha, score_hash_mult, columns)
    cost_alpha_vec = _alpha_vector(cost_alpha, cost_hash_mult, columns)

    # exp()로 log-cost를 원래 스케일로 되돌리면 Jensen 부등식 때문에 구조적으로
    # 과소추정된다(E[exp(residual)] > exp(E[residual]) = 1, residual이 정확히
    # 0으로 안 맞아떨어지는 한). Duan(1983) smearing estimator: 재학습 없이
    # out-of-fold 잔차의 exp() 평균을 모델별로 곱해서 이 편향을 보정한다.
    cost_oof = oof_predictions(matrix, cost_targets, folds=5, alpha=cost_alpha_vec)
    cost_smear = np.exp(cost_targets - cost_oof).mean(axis=0)
    print(
        "cost smearing 보정계수(모델별, out-of-fold): "
        + "  ".join(f"{m}={s:.4f}" for m, s in zip(MODEL_IDS, cost_smear))
    )

    score_mean, score_scale, score_intercept, score_coefficients = fit_ridge(
        matrix, score_targets, score_alpha_vec
    )
    cost_mean, cost_scale, cost_intercept, cost_coefficients = fit_ridge(
        matrix, cost_targets, cost_alpha_vec
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
