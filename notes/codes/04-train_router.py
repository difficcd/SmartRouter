# 원본: training/train_router.py (신규 파일, 전체 그대로)
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

# no quality assurance, 린터에게 규칙위반 직접 고지해서 skip하도록 함
# E402 == pycodestyle 규칙 번호, import 는 원래 최상단인데, src/경로 문제 때문에
# 위의 REPO_ROOT 인자에서 경로 처리를 한 이후에 import(구현한 모듈)해야 함.

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

""" RoutingPolicy : src/ossp_router/resources/routing-policy.v1.json)

"models": {
  "ax31-light":  {"fixed_cost": "0", "input_token_rate": "1",     "output_token_rate": "4"},
  "ax31":        {"fixed_cost": "0", "input_token_rate": "2.127", "output_token_rate": "8.509"},
  "axk1-think":  {"fixed_cost": "0", "input_token_rate": "6.565", "output_token_rate": "26.260"}
},
"tiers": {
  "fast":     {"budget_multiplier": "1.25", "weight": "0.4"},
  "balanced": {"budget_multiplier": "2.0",  "weight": "0.3"},
  "premium":  {"budget_multiplier": "4.0",  "weight": "0.3"}
}

그대로 가져다 사용하는 것.
티어별 budget, model별 token rate 등 들어있음.
(요금표+등급규칙 모음 객체)
"""
""" InputBatch, OutcomeBatch
class Episode:       episode_id, prompt(선택), messages(선택)
class InputBatch:     schema_version, challenge_id, split, episodes: Episode들의 튜플
(문항 여러 개를 하나로 묶어 input 배치 만든게 InputBatch. Episode 를 여러개 묶음.)

class Outcome:        episode_id, model_id, score, num_generations, input_tokens, output_tokens
class OutcomeBatch:    schema_version, challenge_id, split, outcomes: Outcome들의 튜플

---

inputs.episodes  = [Episode("ep-001", prompt="리스트 정렬해줘"), Episode("ep-002", ...), ...]
InputBatch :
{
  "schema_version": 1, "challenge_id": "...", "split": "train",
  "episodes": [
    {"episode_id": "train-0001", "prompt": "Round -63865955 to the nearest one hundred thousand."},
    ... (총 1,760개)
  ]
}

outcomes.outcomes = [
  Outcome("ep-001", "ax31-light", score=0.7, input_tokens=50, output_tokens=120),
  Outcome("ep-001", "ax31",       score=0.9, input_tokens=50, output_tokens=95),
  Outcome("ep-001", "axk1-think", score=0.85, input_tokens=50, output_tokens=310),
  Outcome("ep-002", "ax31-light", ...),
  ...
]

OutcomeBatch :
{
  "episode_id": "train-0001",
  "models": {
    "ax31":       {"input_tokens": 110, "output_tokens": 229, "score": "0.5", "num_generations": 2},
    "ax31-light": {"input_tokens": 112, ...},
    "axk1-think": {...}
  }
}
=> 문항 하나 안에 모델 3개 결과가 묶여있는 모양임.ㄱ
parse_outcomes 함수 참조 : 원본은 문항별로 묶은 모양 => 파싱 => outcom튜플에 담기

n차원을 1차원 데이터로 flatten하는 느낌..
원본 json을 flatten 해서 Outcome 객체 여러개로 만들고(문항x모델 =개별 결과 하나하나, 5280개)
이걸 묶어서 OutcomeBatch로 하여 부가정보추가+Outcome 을 배치로 관리하는 것. 


inputs.episodes는 문항 1,760개짜리 목록,
outcomes.outcomes는 문항×모델 3개 = 5,280개짜리 평평한(flat) 목록.

"""


def _outcome_cost(outcome, policy: RoutingPolicy) -> float:
    # policy.v1.schema ref.

    rates = policy.models[outcome.model_id]     # 모델을 사용할 때 매겨지는 비용 단가표
    unit = Decimal(policy.token_unit)           # 토큰 계산 기준 단위 (예: 1000토큰 기준 등)

    # models[req]=>rates mapping : fixed_cost, input_tokens, output_tokens
    # policy.v1.schema.json : "$ref": "#/$defs/rates"

    cost = (
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )
    value = float(cost) 
    # 계산 시에는 오차 방지했으니 float 형변환해도 오차 없음. 
    # ML lib에 넣기 위해 float 형변환

    # input_token_rate, output_token_rate : rates -properties "#/$defs/decimalString"
    # cost = fixed + input token score (토큰수*rate/기준단위) + output token score

    if not math.isfinite(value) or value <= 0: # 오버플로, NaN 등 계산오류 방지.
        raise ValueError("학습 outcome의 모델 비용은 0보다 커야 합니다.")
    return value
# 문항 id별 각 모델의 outcome 객체를 묶은 list를 outcome으로 받고, RoutingPolicy 받아서
# 모델의 총 비용을 합산해서(cost 총 합산) float value로 return



def build_training_matrix(
    inputs: InputBatch, outcomes: OutcomeBatch, policy: RoutingPolicy, hash_bins: int
) -> Tuple[Any, Any]:

    # (episode id, model id) : Outcome()
    outcome_index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    """outcomes.outcomes = [
    Outcome("ep-001", "ax31-light", score=0.7, input_tokens=50, output_tokens=120),
    Outcome("ep-001", "ax31",       score=0.9, input_tokens=50, output_tokens=95),
    Outcome("ep-001", "axk1-think", score=0.85, input_tokens=50, output_tokens=310),
    Outcome("ep-002", "ax31-light", ...),
    ...
    ]

    =>
    {
        ("ep1", "ax31"): Outcome(...),
        ("ep1", "ax31-light"): Outcome(...),
        ("ep1", "axk1-think"): Outcome(...),
        ("ep2", "ax31"): Outcome(...),
        ...
    } ('문항별 모델결과' 매핑, 이렇게 해야 쉽게 찾을 수 있음)
    
    """

    expected = {
        (episode.episode_id, model_id)
        for episode in inputs.episodes
        for model_id in MODEL_IDS
    } 
    # set 구성. 바깥 for문에서는 episodes 돌고 안쪽은 모델 3개 돌아서
    # 1,760×3개의 (episode_id, model_id) 짝 세트 == 문항,모델 짝 집합
    # expected == inputs.json, MODEL_IDS(코드 상수) 에서 만들어낸 값.
    # 위의 outcomes 랑 직접적인 관계 없음


    if set(outcome_index) != expected:
        raise ValueError("Train outcome 행렬이 입력과 모델 전체를 포함하지 않습니다.")
    # 일치하지 않는 경우를 방지. (데이터 결실, 처리 실패 등)

    matrix = np.asarray(
        [_team_raw_feature_vector(episode, hash_bins) for episode in inputs.episodes],
        dtype=np.float64,
    )
    # _team_raw_feature_vector가 뱉은 266개 float 튜플들을 묶어서 array로 만들어 행렬화
    # dtype == data type (float64 == 배정도 부동소수점, double precision)

    targets = []
    
    for episode in inputs.episodes:
        rows = [outcome_index[(episode.episode_id, m)] for m in MODEL_IDS]
        # 문항별 각 모델의 outcome() 객체 list

        scores = [float(row.score) for row in rows]
        # row에서 모델의 score list

        log_costs = [math.log(_outcome_cost(row, policy)) for row in rows]
        # row의 모델별 log(cost) list
        # 비용 편차가 클 때를 대비해 log 스케일링하여 안정성 높임

        targets.append(scores + log_costs) # [scroes list, log_costs]

        # episode 별 모델들의 결과를 수집하는 것
        # scores: [모델A점수, 모델B점수, 모델C점수]
        # log_costs: [모델A비용, 모델B비용, 모델C비용]

    return matrix, np.asarray(targets, dtype=np.float64) # targets도 matrix화해서 배정도로 return)
# episode 내용 임베딩해서 특성 추출한 후 matrix화하여 return 
# 문항별 score, log(cost) 합쳐(list를 이어서.) matrix화하여 return 




def fit_ridge(matrix, targets, alpha: float):
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    intercept = targets.mean(axis=0)
    centered = targets - intercept

    rows, columns = standardized.shape
    if rows <= columns:
        system = standardized @ standardized.T + alpha * np.eye(rows)
        coefficients = standardized.T @ np.linalg.solve(system, centered)
    else:
        system = standardized.T @ standardized + alpha * np.eye(columns)
        coefficients = np.linalg.solve(system, standardized.T @ centered)
    return mean, scale, intercept, coefficients


def predict_ridge(matrix, mean, scale, intercept, coefficients):
    return (matrix - mean) / scale @ coefficients + intercept


def oof_predictions(matrix, targets, *, folds: int, alpha: float):
    rows = matrix.shape[0]                # 전체 데이터(episodes) 개수. 셰잎의 [0]
    predictions = np.empty_like(targets)  # targets의 형식 가진 빈(초기화X) 배열을 생성
    fold_ids = np.arange(rows) % folds    # 데이터 개수만큼 array 만들기 (100이면 0~99) % 데이터 쪼개기

    for fold in range(folds):
        validation = fold_ids == fold   # flod_ids = ndarray
        training = ~validation          # 
        mean, scale, intercept, coefficients = fit_ridge(matrix[training], targets[training], alpha)
        predictions[validation] = predict_ridge(matrix[validation], mean, scale, intercept, coefficients)
    return predictions
# Out Of Fold == 모델 검증할 때 쓰는 교차검증 기법과 유사
# OOF : 실제로 이 모델이 처음 보는 데이터를 얼마나 잘 맞추는지 평가하는 것
# folds == 데이터를 쪼갤 단위


def select_alpha_single(matrix, targets, *, folds: int, candidates: Sequence[float], label: str) -> float:
    """targets here is ONE block (score-only or log-cost-only) -- fit and pick
    alpha for that block alone. A shared alpha across score+cost let score's
    much larger MSE dominate the objective and left the cost heads
    under-regularized."""

    best_alpha = candidates[0]   # 0.1~10^6 가중치 list
    best_mse = math.inf          # 무한대의 숫자 (오차니까 무한대로 초기화)

    for alpha in candidates:
        predictions = oof_predictions(matrix, targets, folds=folds, alpha=alpha)
        mse = float(np.mean((predictions - targets) ** 2))
        print(f"  [{label}] alpha={alpha:>10.4g}  mse={mse:.5f}")

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    return best_alpha


# ==== main ==== # 

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-outcomes", type=Path, required=True)
    parser.add_argument("--hash-bins", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    policy = load_bundled_policy() # 내장 정책 파일 읽어오기. (RoutingPolicy)

    inputs = load_input(args.train_input)
    outcomes = load_outcomes(args.train_outcomes)

    # protocol의 parse_* 까지 이해할 필요는 없고,
    # protocol에서 던져주는 Type의 꼴만 이해하면 됨. (dataclass + json형식)

    print(f"학습 문항 수: {len(inputs.episodes)}  hash_bins={args.hash_bins}")
    matrix, targets = build_training_matrix(inputs, outcomes, policy, args.hash_bins)

    print(f"특징 행렬: {matrix.shape}  타깃: {targets.shape}")
    # build_training_matrix  matrix(특징행렬), targets(model별 log_cost+scores)

    model_count = len(MODEL_IDS)                # 모델 수
    score_targets = targets[:, :model_count]    
    cost_targets = targets[:, model_count:]   

    # scores: [모델A점수, 모델B점수, 모델C점수]
    # log_costs: [모델A비용, 모델B비용, 모델C비용]



    candidates = [10.0**exponent for exponent in range(-1, 7)]
    # 0.1~10^6 : 규제 강도 candidates

    print("alpha 후보 탐색 -- score와 cost를 따로 고른다 (out-of-fold, 5-fold):")
    score_alpha = select_alpha_single(matrix, score_targets, folds=5, candidates=candidates, label="score")
    cost_alpha = select_alpha_single(matrix, cost_targets, folds=5, candidates=candidates, label="cost ")

    print(f"선택된 alpha: score={score_alpha:g}  cost={cost_alpha:g}")

    score_mean, score_scale, score_intercept, score_coefficients = fit_ridge(
        matrix, score_targets, score_alpha
    )
    cost_mean, cost_scale, cost_intercept, cost_coefficients = fit_ridge(
        matrix, cost_targets, cost_alpha
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
        "tier_safety_ratios": {tier: 1.0 for tier in TIERS},  # placeholder -- calibrate next
        "risk_multiplier": {model_id: 1.0 for model_id in MODEL_IDS},  # placeholder
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
