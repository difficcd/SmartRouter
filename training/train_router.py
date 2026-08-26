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
    _TEAM_ARTIFACT_TYPE,
    _TEAM_FEATURE_VERSION,
    _TEAM_HASH_ALGORITHM,
    _TEAM_SCHEMA_VERSION,
    _TEAM_DENSE_FEATURE_NAMES,
    _team_expand_basis,
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
) -> Tuple[Any, Any, Any, Any]:
    outcome_index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    expected = {
        (episode.episode_id, model_id)
        for episode in inputs.episodes
        for model_id in MODEL_IDS
    }
    if set(outcome_index) != expected:
        raise ValueError("Train outcome 행렬이 입력과 모델 전체를 포함하지 않습니다.")

    raw = np.asarray(
        [_team_raw_feature_vector(episode, hash_bins) for episode in inputs.episodes],
        dtype=np.float64,
    )
    # The trigonometric basis is standardized against Train's own dense block,
    # and the router has to reproduce that exactly, so these two vectors ride
    # along in the artifact. They are NOT the same as feature_mean/scale, which
    # standardize the whole 326-wide matrix after expansion.
    dense = len(_TEAM_DENSE_FEATURE_NAMES)
    basis_mean = raw[:, :dense].mean(axis=0)
    basis_scale = raw[:, :dense].std(axis=0)
    basis_scale = np.where(basis_scale > 1e-12, basis_scale, 1.0)
    matrix = np.asarray(
        [_team_expand_basis(row, basis_mean, basis_scale) for row in raw],
        dtype=np.float64,
    )
    targets = []
    for episode in inputs.episodes:
        rows = [outcome_index[(episode.episode_id, m)] for m in MODEL_IDS]
        scores = [float(row.score) for row in rows]
        log_costs = [math.log(_outcome_cost(row, policy)) for row in rows]
        targets.append(scores + log_costs)
    return matrix, np.asarray(targets, dtype=np.float64), basis_mean, basis_scale


# Bagging is module state rather than a parameter threaded through every call
# site, and that is deliberate. fit_ridge is reached from four places -- alpha
# selection, out-of-fold predictions, the smearing correction, and the final
# fit -- and if any one of them disagreed with the others about whether to bag,
# the artifact would be an inconsistent mixture with nothing to warn us. One
# switch makes that impossible.
_BAGS = 0
_BAG_SEED = 20260823


def _solve_ridge(standardized, centered, alpha: float):
    rows, columns = standardized.shape
    if rows <= columns:
        system = standardized @ standardized.T + alpha * np.eye(rows)
        return standardized.T @ np.linalg.solve(system, centered)
    system = standardized.T @ standardized + alpha * np.eye(columns)
    return np.linalg.solve(system, standardized.T @ centered)


def fit_ridge(matrix, targets, alpha: float):
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    intercept = targets.mean(axis=0)
    centered = targets - intercept

    if _BAGS <= 0:
        return mean, scale, intercept, _solve_ridge(standardized, centered, alpha)

    # Standardization and intercept come from the FULL block, not from each
    # resample, so the averaged coefficients remain one usable coefficient set.
    # Bagging has to stay free at inference or it cannot ship in the container.
    rng = np.random.default_rng(_BAG_SEED)
    accumulated = np.zeros((standardized.shape[1], centered.shape[1]))
    for _ in range(_BAGS):
        pick = rng.integers(0, standardized.shape[0], standardized.shape[0])
        accumulated += _solve_ridge(standardized[pick], centered[pick], alpha)
    return mean, scale, intercept, accumulated / _BAGS


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


def select_cost_alpha(matrix, targets, *, folds: int, candidates: Sequence[float]) -> float:
    """Like select_alpha_single, but scores candidates in REAL cost space
    (exp of the log-cost target) instead of log space.

    The router and the official grader both work in real cost -- the budget
    check is a sum of real credits. Picking alpha by log-space MSE optimizes
    a quantity nobody uses: measured in real space, log-optimal alpha=1000
    gives per-episode correlation 0.56/0.60/0.18 (light/ax31/axk1), while
    alpha=1e4 gives 0.62/0.65/0.22 with lower real-space error too. Better
    real-space cost accuracy is the binding constraint on how much budget a
    tier can safely spend, so this is measured where it matters.

    Each candidate's predictions are batch-calibrated before scoring (the
    same correction main() applies), so candidates are compared on shape
    rather than on level.
    """
    real = np.exp(targets)
    best_alpha = candidates[0]
    best_error = math.inf
    for alpha in candidates:
        predictions = np.exp(oof_predictions(matrix, targets, folds=folds, alpha=alpha))
        predictions = predictions * (real.sum(axis=0) / predictions.sum(axis=0))
        error = float(np.mean((predictions - real) ** 2))
        print(f"  [cost ] alpha={alpha:>10.4g}  real-space mse={error:.6e}")
        if error < best_error:
            best_error = error
            best_alpha = alpha
    return best_alpha


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-outcomes", type=Path, required=True)
    parser.add_argument("--hash-bins", type=int, default=256)
    parser.add_argument(
        "--cost-fit",
        choices=("split", "classic"),
        default="split",
        help="split: 예산용/랭킹용 alpha를 따로 + 배치합 보정 (v10/v11). "
             "classic: 하나의 log-MSE alpha + Duan smearing (v8).",
    )
    parser.add_argument(
        "--bags",
        type=int,
        default=0,
        help="0이면 단일 적합. >0이면 Train 행을 이 횟수만큼 재표본해 계수를 "
             "평균낸다(배깅). 계수는 여전히 한 벌이므로 추론 비용은 0. "
             "홀드아웃 기준 300회부터 시드 영향이 무의미해진다.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    global _BAGS
    _BAGS = args.bags
    if _BAGS > 0:
        print(f"배깅 {_BAGS}회 (시드 {_BAG_SEED}) -- alpha 선택, out-of-fold, "
              f"smearing, 최종 적합 전부 동일 설정")

    policy = load_bundled_policy()
    inputs = load_input(args.train_input)
    outcomes = load_outcomes(args.train_outcomes)

    print(f"학습 문항 수: {len(inputs.episodes)}  hash_bins={args.hash_bins}")
    matrix, targets, basis_mean, basis_scale = build_training_matrix(
        inputs, outcomes, policy, args.hash_bins
    )
    print(f"특징 행렬: {matrix.shape}  타깃: {targets.shape}")

    model_count = len(MODEL_IDS)
    score_targets = targets[:, :model_count]
    cost_targets = targets[:, model_count:]

    candidates = [10.0**exponent for exponent in range(-1, 7)]
    print("alpha 후보 탐색 -- score와 cost를 따로 고른다 (out-of-fold, 5-fold):")
    score_alpha = select_alpha_single(matrix, score_targets, folds=5, candidates=candidates, label="score")

    # The cost prediction does two different jobs and they want opposite
    # regularization, so fit it twice:
    #
    #   BUDGET  -- "what will this batch cost in total?" Feeds cap and the
    #     official budget check, both of which live in real (exp) space.
    #     Chosen by real-space error; lands on very heavy regularization.
    #   RANKING -- "which episode is the best value to promote?" Feeds the
    #     EV comparison, which only cares about the ratio gain/cost, i.e.
    #     relative differences between episodes. Chosen by log-space MSE
    #     (log error == relative error); lands much lighter.
    #
    # Measured on Train (fast-tier greedy with perfect cost accounting, so
    # only ranking quality varies): alpha=300 -> 0.6423, 1000 -> 0.6382,
    # 1e5 -> 0.6213. Meanwhile real-space error keeps improving all the way
    # to 1e5. v10 used the real-space alpha for both and won premium (budget
    # bound) while losing fast (ranking bound); splitting them takes both.
    #
    # --cost-fit selects between the two families measured so far:
    #   split   (v10/v11) -- budget head by real-space error, ranking head by
    #                        log MSE, batch-sum level correction on each.
    #   classic (v8)      -- one alpha by log MSE for both heads, Duan
    #                        per-episode smearing. Both heads identical, which
    #                        is exactly the single-head model v8 shipped.
    # v8 still scores higher than v11/v12 on BOTH public splits, so "split"
    # is not established as better once safety is calibrated properly; this
    # flag exists to compare them under identical calibration.
    if args.cost_fit == "split":
        cost_budget_alpha = select_cost_alpha(matrix, cost_targets, folds=5, candidates=candidates)
        cost_rank_alpha = select_alpha_single(
            matrix, cost_targets, folds=5, candidates=candidates, label="c-rank"
        )
    else:
        cost_budget_alpha = select_alpha_single(
            matrix, cost_targets, folds=5, candidates=candidates, label="cost "
        )
        cost_rank_alpha = cost_budget_alpha
    print(
        f"선택된 alpha({args.cost_fit}): score={score_alpha:g}  "
        f"cost(예산)={cost_budget_alpha:g}  cost(랭킹)={cost_rank_alpha:g}"
    )

    # exp()로 log-cost를 원래 스케일로 되돌리면 Jensen 부등식 때문에 구조적으로
    # 과소추정된다. v1은 Duan(1983) smearing(잔차 exp()의 문항별 평균)으로 이걸
    # 보정했는데, 실측해보니 그 보정이 배치 합계 기준으로는 크게 과보정이었다
    # (Train에서 예측합/실제합이 light 1.35, ax31 1.24, axk1 1.13).
    #
    # 중요한 건 문항별 기댓값이 아니라 배치 합계다 -- 라우터의 cap도, 공식
    # 채점의 예산 검사도 전부 "이 배치 전체 비용의 합"만 본다. 합이 부풀려지면
    # 안전계수를 과도하게 조여야 하고(fast는 1.25배 예산 중 실제로 1.025배만
    # 쓰게 됐다), 그만큼 승격 여지를 통째로 날린다.
    #
    # 그래서 보정계수를 "out-of-fold 예측 합이 실제 합과 일치하도록" 직접
    # 잡는다. Duan smearing이 문항별 불편추정을 노린다면 이건 배치합 불편추정을
    # 노리는 것 -- 우리 목적함수에 정확히 맞는 쪽.
    # Each head gets its own level correction, since each one's raw exp()
    # level is off by a different amount.
    def level_correction(alpha):
        oof = oof_predictions(matrix, cost_targets, folds=5, alpha=alpha)
        if args.cost_fit == "split":
            return np.exp(cost_targets).sum(axis=0) / np.exp(oof).sum(axis=0)
        return np.exp(cost_targets - oof).mean(axis=0)  # Duan smearing (v8)

    cost_smear = level_correction(cost_budget_alpha)
    cost_rank_smear = (
        cost_smear if cost_rank_alpha == cost_budget_alpha
        else level_correction(cost_rank_alpha)
    )
    print(
        "cost 배치합 보정계수 -- 예산용: "
        + "  ".join(f"{m}={s:.4f}" for m, s in zip(MODEL_IDS, cost_smear))
    )
    print(
        "                       랭킹용: "
        + "  ".join(f"{m}={s:.4f}" for m, s in zip(MODEL_IDS, cost_rank_smear))
    )

    score_mean, score_scale, score_intercept, score_coefficients = fit_ridge(
        matrix, score_targets, score_alpha
    )
    cost_mean, cost_scale, cost_intercept, cost_coefficients = fit_ridge(
        matrix, cost_targets, cost_budget_alpha
    )
    rank_mean, rank_scale, rank_intercept, rank_coefficients = fit_ridge(
        matrix, cost_targets, cost_rank_alpha
    )
    assert np.allclose(score_mean, cost_mean) and np.allclose(score_scale, cost_scale)
    assert np.allclose(score_mean, rank_mean) and np.allclose(score_scale, rank_scale)
    mean, scale = score_mean, score_scale

    def head_dict(intercept: float, coefficients) -> dict:
        return {"intercept": float(intercept), "coefficients": [float(c) for c in coefficients]}

    artifact = {
        # Taken from heuristic.py rather than repeated here. These four fields
        # exist so a mismatch between artifact and code is detectable, and a
        # literal copy in the trainer defeats that: bumping the feature version
        # in one place and not the other produces an artifact that fails its own
        # check, which is what just happened.
        "artifact_type": _TEAM_ARTIFACT_TYPE,
        "schema_version": _TEAM_SCHEMA_VERSION,
        "feature_version": _TEAM_FEATURE_VERSION,
        "hash_algorithm": _TEAM_HASH_ALGORITHM,
        "hash_bins": args.hash_bins,
        "dense_feature_names": list(_TEAM_DENSE_FEATURE_NAMES),
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "basis_mean": [float(v) for v in basis_mean],
        "basis_scale": [float(v) for v in basis_scale],
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
        "log_cost_rank_heads": {
            model_id: head_dict(rank_intercept[i], rank_coefficients[:, i])
            for i, model_id in enumerate(MODEL_IDS)
        },
        "cost_rank_smear": {
            model_id: float(cost_rank_smear[i]) for i, model_id in enumerate(MODEL_IDS)
        },
        "tier_safety_ratios": {tier: 1.0 for tier in TIERS},  # placeholder -- calibrate next
        # placeholder -- per-tier now, since fast/premium want opposite values
        "risk_multiplier": {
            tier: {model_id: 1.0 for model_id in MODEL_IDS} for tier in TIERS
        },
        "high_cap_ratio": {tier: 1.0 for tier in TIERS},  # placeholder -- calibrate next
        "share_ratio": {tier: 1.0 for tier in TIERS},  # placeholder -- calibrate next
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
