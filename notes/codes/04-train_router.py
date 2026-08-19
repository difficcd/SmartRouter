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
    # matrix.shape : (1760, 266) == (episode 개수, feature 개수(hash+fixed))

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

    # aixs=0:row기준으로, row방향 따라 게산 : matrix[0] == 결과 개수

    mean = matrix.mean(axis=0)                      # all episode 임베딩값의 mean
    scale = matrix.std(axis=0)                      # all episode 임베딩값의 표준편차
    scale = np.where(scale > 1e-12, scale, 1.0)     
    # 1e-12(0근사 미세한값)<=std인 경우 std=1로 처리.  : 0으로 나누기 오류 방지 
    # 같은 특징끼리만(n번째 특징끼리) 1,760개별 평균 도출 (shape: (feature_cnt,))

    standardized = (matrix - mean) / scale       
    # 알파 가중치별 ridge 회귀 정답값 분포를 보고 싶은건데,
    # 데이터 노이즈 때문에 결과에 영향을 미치면 안 되므로
    # ridge 핵심인 alpha영향력을 보기 위한 과정, 배치 정규화와 유사함

    intercept = targets.mean(axis=0)
    centered = targets - intercept
    # targets는 정규화 대신 중앙화(centering)
    # 절편을 계산에서 분리하기 위한 편의상 조건
    # 원점이 데이터의 mean에 위치해있어야 더 깔끔하고 오차 줄여줌. + bias, 노이즈 빼기

    rows, columns = standardized.shape

    if rows <= columns: # 연산 최적화 (행렬곱 최적화) : 샘플 수 <= 특성 수 

        system = standardized @ standardized.T + alpha * np.eye(rows)
        coefficients = standardized.T @ np.linalg.solve(system, centered)
        # linalg == linear algebra module(numpy), solve(A, b) == Ax=b 해 반환.
        # system*x = centered 의 x를 구함.
        # coefficients == 희귀계수 (matrix*alpha* "x" = targets)

    else:
        system = standardized.T @ standardized + alpha * np.eye(columns)
        coefficients = np.linalg.solve(system, standardized.T @ centered)

    # @ == 행렬곱 연산자. np.dot(A, B) == A @ B
    # matrix : ndarray.T ==  Transpose 행렬

    return mean, scale, intercept, coefficients
# 각 알파값별로 데이터를 임의로 뽑아 테스트하는 oof helper
#    알파값별로 가중치를 알아내서, 규제강도(alpha) 조절했을 때 가중치(coeff=x) 변화량 return
# train data의 matrix(특징행렬), targets(score+cost) 받아서 알파값(가중치) 적용

# return : mean(matrix episode임베딩값의 mean),  scale(std),
#          intercept(targets mean, 절편), coeff(x=weight)



def predict_ridge(matrix, mean, scale, intercept, coefficients):
    return (matrix - mean) / scale @ coefficients + intercept
# 훈련때 구해둔 mean, std, intercept, coeff 그대로 가져와서 test용에다 넣음
# 검증용 데이터 정규화한 다음 가중치 곱하고 intercept(빼뒀던 mean) 더해서 최종 예측점수 return 



def oof_predictions(matrix, targets, *, folds: int, alpha: float):
    # folds == 데이터를 쪼갤 단위
    # k겹 교차검증 (k fold cross-validation). 인공지능_1 17p 참조


    rows = matrix.shape[0]                # 전체 데이터(episodes) 개수. 셰잎의 [0]
    predictions = np.empty_like(targets)  # targets의 형식 가진 빈(초기화X) 배열을 생성
    fold_ids = np.arange(rows) % folds    # 데이터 개수 ndarray (100이면 0~99) % 데이터 쪼개기

    for fold in range(folds):
        validation = fold_ids == fold   # 현 시점에서의 test data (임의)
        training = ~validation          # 현 시점에서의 train data 

        mean, scale, intercept, coefficients = fit_ridge(matrix[training], targets[training], alpha)
        # matrix[training] == Boolean indexing. training은 Boolean ndarray (data개수만큼 존재)
        # ndarray[*] 는, *라는 Boolean ndarray 에 따라 ndarray 원본값 중 true인 위치에서만 뽑아라.
        # 결국 여기서의 mean, scale .. coeff = alpha값 적용한 weight의 변화를 알아보기 위한 변수들.

        predictions[validation] = predict_ridge(matrix[validation], mean, scale, intercept, coefficients)
        # 현재 test (validation용) 데이터들에 대한 예측값 뽑아내기

    return predictions
# Out Of Fold == 모델 검증할 때 쓰는 교차검증 기법과 유사
# OOF : 실제로 이 모델이 처음 보는 데이터를 얼마나 잘 맞추는지 평가하는 것
# 현재 받아온 알파값이 학습에서의 가중치에 얼마나 영향을 주는지를 알기 위해 
# 임의의 train, test데이터 만들고 학습시킨 다음 test데이터에 대한 결과값을 return

# 이 모델이 특정 알파값에서 얼마나 똑똑한지, 혹은 얼마나 과적합(Overfitting)되지 않는지를 
# 수치로 확인하기 위해 전 데이터를 한 번씩 다 시험해 보는 과정



def select_alpha_single(matrix, targets, *, folds: int, candidates: Sequence[float], label: str) -> float:
    """targets here is ONE block (score-only or log-cost-only) -- fit and pick
    alpha for that block alone. A shared alpha across score+cost let score's
    much larger MSE dominate the objective and left the cost heads
    under-regularized."""
    # score / log_cost 분리해서 순회하는 이유 ref.

    best_alpha = candidates[0]   # 0.1~10^6 가중치 list
    best_mse = math.inf          # 무한대의 숫자 (오차니까 무한대로 초기화)

    for alpha in candidates:     
        predictions = oof_predictions(matrix, targets, folds=folds, alpha=alpha)
        # 현재 알파값에 대해 oof predict 수행

        mse = float(np.mean((predictions - targets) ** 2))
        print(f"  [{label}] alpha={alpha:>10.4g}  mse={mse:.5f}")

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    return best_alpha
# oof predict 에서의 predict와 targets 비교:
# MSE가 가장 작은 alpha값이 beat_alpha값으로 선택, return





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
    # 숫자의 분포/범위가 다르며, 특성이 다르기 때문에 따로 처리해야 함 (합쳐서 보면 cost가 압도적)

    print(f"선택된 alpha: score={score_alpha:g}  cost={cost_alpha:g}")



    score_mean, score_scale, score_intercept, score_coefficients = fit_ridge(
        matrix, score_targets, score_alpha
    )
    cost_mean, cost_scale, cost_intercept, cost_coefficients = fit_ridge(
        matrix, cost_targets, cost_alpha
    )
    assert np.allclose(score_mean, cost_mean) and np.allclose(score_scale, cost_scale)
    mean, scale = score_mean, score_scale
    # assert == 반드시 참인 조건을 확인함 (False 이면 프로그램 죽임)
    # allclose() == 두 개의 배열이 소수점 오차 범위 내에서 거의 비슷한가?를 비교해 bool return
    # 입력 데이터가 완전히 동일한지 교차 검증하는 것. (전처리 무결성 보장)
    # matrix라는 동일 행렬로부터 나온 mean, std이므로, 동일해야만 함.
    # mean, scale은 matrix 총 기준이고, intercept/coeffi만 다름

    def head_dict(intercept: float, coefficients) -> dict:
        return {"intercept": float(intercept), "coefficients": [float(c) for c in coefficients]}
    # "score_heads": { "ax31": { "coefficients": [], "intercept": }, "ax31-light": {}, "axk1-think": {} }

    # artifact.json 구성 (tier_safety_ratios, risk_multiplier 두가지는 임시값)
    # training/artifact.json 구조 참조.
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
        # 모델 정책이 "단 하나라도 수정되었는지(무결성 검증)"를 판별

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
        # enumerate 의 i값은 intercept, coeffic 뽑아내기 위해서만 사용

        "tier_safety_ratios": {tier: 1.0 for tier in TIERS},  # placeholder -- calibrate next
        "risk_multiplier": {model_id: 1.0 for model_id in MODEL_IDS},  # placeholder
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # out 파일경로 parent 를 대상으로 mkdir (sibling) : --out args
    # parents=True: 중간에 폴더가 없으면 부모 폴더들까지 만들어 줌
    # exist_ok=True: 파일이 이미 존재해도 덮어쓰기

    args.out.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    ) 
    # artifact 를 json 문자열로 변경, 유니코드 ASCII로 저장하지 않고 원래 글자 그대로 출력(False)
    # indent=2 들여쓰기 2칸, 가독성을 위한 것.   sort_keys : 딕셔너리 키들을 알파벳순 정렬
    # SHA-256 해시를 일관되게 만들기 위한 필수 조건. 

    print(f"OK: {args.out}에 학습 파일을 저장했습니다.")

    print("주의: tier_safety_ratios/risk_multiplier는 아직 자리표시자(1.0)입니다.")
    print("      training/calibrate_safety.py -> training/bake_artifact.py 순으로 이어서 실행할 것.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
