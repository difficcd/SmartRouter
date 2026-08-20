# 원본: training/calibrate_safety.py (신규 파일, 전체 그대로)
# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Pick tier_safety_ratios and risk_multiplier from bootstrap overrun probability,
not from maximizing Dev score. This is the fix for the exact failure mode
`baselines/hash_regex.py` had (Premium passed Dev at 99.6% of the limit,
then blew past it on the hidden eval set).

Method (per tier):
  1. Build predicted and real (score, cost) matrices for Dev once.
  2. For a grid of (risk_multiplier[axk1-think], safety_ratio):
     bootstrap-resample Dev episodes (with replacement) K times; for each
     resample, re-run the SAME batch-level allocation the router will run
     (light_total and λ are recomputed per resample, since that's what an
     official grader would compute for a differently-composed hidden batch)
     and measure the REALIZED budget ratio and tier score against the real
     outcomes.
  3. Reject any (risk_multiplier, safety_ratio) whose bootstrap overrun
     probability exceeds --overrun-target; among what's left, keep the one
     with the highest mean tier score.
  4. risk_multiplier is shared across tiers (it's a property of how much we
     trust the model's cost prediction, not of the budget); safety_ratio is
     per-tier. Pick the one risk_multiplier that maximizes the *weighted*
     score across all three tiers.

The inner allocation loop is vectorized with NumPy for speed (training-only;
the shipped src/team_router/allocate.py stays pure Python). Uses the exact
official cost formula so bootstrap numbers are directly comparable to
self-check output.

    py -3.13 training/calibrate_safety.py \
        --artifact src/team_router/resources/router.v1.json \
        --dev-input data/materialized/dev/inputs.json \
        --dev-outcomes data/dev/outcomes.json \
        --train-input data/materialized/train/inputs.json \
        --train-outcomes data/train/outcomes.json \
        --out src/team_router/resources/router.v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ossp_router.heuristic import _team_parse_artifact, _team_predict_episode  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    InputBatch,
    OutcomeBatch,
    RoutingPolicy,
    load_bundled_policy,
    load_input,
    load_outcomes,
)


def load_artifact(path: Path):
    return _team_parse_artifact(path.read_text(encoding="utf-8"))
# artifact 파일 utf-8로 인코딩한후 읽고 파싱해서 return 

def predict_batch(episodes, artifact):
    scores, costs = [], []

    for episode in episodes:
        episode_scores, episode_costs = _team_predict_episode(episode, artifact)
        scores.append(episode_scores)
        costs.append(episode_costs)
        
    return scores, costs
# return  batch predict value : scores, costs  (list)


RNG_SEED = 20260815
BOOTSTRAP_K = 300
OVERRUN_TARGET = 0.01
SAFETY_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
RISK_HIGH_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]



def _real_cost(outcome, policy: RoutingPolicy) -> float:
    rates = policy.models[outcome.model_id]
    # policy schema, routing policy.v1 json 참조

    unit = Decimal(policy.token_unit)
    return float(
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )
# out, policy 받아서 실제 비용 계산 (fixed+outcome 반영한 in.out 총 token)



def build_matrices(
    inputs: InputBatch, outcomes: OutcomeBatch, policy: RoutingPolicy, artifact
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (predicted_scores, predicted_costs, real_scores, real_costs), each (epi_cnt, model_cnt)."""

    predicted_scores, predicted_costs = predict_batch(inputs.episodes, artifact)
    # 현재 weight (matrix) 에 대한 predict값 (실제 score, costs)

    outcome_index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    # (epi_id, model_id) : outcome()

    n = len(inputs.episodes)

    n_models = len(MODEL_IDS)               # 하드코딩 수정 
    pred_scores = np.empty((n, n_models), dtype=np.float64)
    pred_costs = np.empty((n, n_models), dtype=np.float64)
    real_scores = np.empty((n, n_models), dtype=np.float64)
    real_costs = np.empty((n, n_models), dtype=np.float64)

    # n행 n_models열 빈(초기화 X) 실수형 넘파이 배열 ndarray


    for row, episode in enumerate(inputs.episodes):
        for col, model_id in enumerate(MODEL_IDS):
            # row = episode, col = model_id

            pred_scores[row, col] = predicted_scores[row][model_id]
            pred_costs[row, col] = predicted_costs[row][model_id]
            # scores, costs list => ndarray

            outcome = outcome_index[(episode.episode_id, model_id)]
            real_scores[row, col] = float(outcome.score)
            real_costs[row, col] = _real_cost(outcome, policy)
            
    return pred_scores, pred_costs, real_scores, real_costs
# in, out, policy, artifact 받아서  scores/costs 의 예측값, 실제값 return 



def allocate_vectorized(
    pred_scores: np.ndarray,
    pred_costs: np.ndarray,
    *,
    budget_multiplier: float,
    safety_ratio: float,
    risk_multiplier: np.ndarray,
    bisection_iterations: int = 60,
) -> np.ndarray:
    """Same rule as team_router.allocate.select_models, vectorized. Returns
    a (n,) int array of chosen column indices (0=light,1=ax31,2=axk1)."""

    light_total = pred_costs[:, 0].sum()
    cap = light_total * max(1.0, budget_multiplier * safety_ratio)
    # 가장 가벼운 모델로만 돌렸을 때의 예상 비용 구하고
    # 예산배율, 안전계수곱해서 이번배치 사용가능한 최대예산 설정 (cap)

    def choose(penalty: float) -> Tuple[np.ndarray, float]:
        ev = pred_scores - penalty * risk_multiplier[None, :] * pred_costs / light_total
        # 가치 평가 eval = 예상점수 - 패널티*risk val*예상 비용 / 가장가벼운모델 예상비용

        choice = np.argmax(ev, axis=1)  # ties -> first (cheapest) column, matches -index tie-break
        # ev가 가장높은 모델 번호 고르기

        total = pred_costs[np.arange(len(choice)), choice].sum() 
        # choice 선택후 총비용 합 계산해 반환

        return choice, total
    # penalty 매겼을 때 각 문항이 어떤 모델 선택하는게 가성비 좋은지 계산

    choice, total = choose(0.0)
    if total > cap:
        low, high = 0.0, 1.0
        choice, total = choose(high)

        while total > cap and high < 2.0**40:
            low, high = high, high * 2.0
            choice, total = choose(high)

        for _ in range(bisection_iterations):
            middle = (low + high) / 2.0
            candidate_choice, candidate_total = choose(middle)
            if candidate_total <= cap:
                high = middle
                choice, total = candidate_choice, candidate_total
            else:
                low = middle
    if total > cap:
        choice = np.zeros(len(pred_scores), dtype=int)
    # 이분 탐색 : penalty=0으로 최적모델 고른뒤 total>cap이면 비용 비쌀수록 penalty
    # fallback : 이분탐색 돌려도 초과 해결 못 하면, np.zeros로 구멍 막기.
    

    return choice
# returns choice : 각 문항별로 선택된 모델의 인덱스 배열 반환.
# 라우팅의 핵심 함수로, 이부분을 개선할 여지가 많음.



def bootstrap_evaluate(
    pred_scores: np.ndarray,
    pred_costs: np.ndarray,
    real_scores: np.ndarray,
    real_costs: np.ndarray,
    *,
    budget_multiplier: float,
    safety_ratio: float,
    risk_multiplier: np.ndarray,
    rng: np.random.Generator,
    k: int = BOOTSTRAP_K,
) -> Tuple[float, float]:
    """Returns (overrun_probability, mean_tier_score) over k bootstrap resamples."""

    n = len(pred_scores)
    overruns = 0
    score_total = 0.0

    for _ in range(k):  
        # 총 k회, pred_scores 의 data로 복원추출 (bootstrap)

        index = rng.integers(0, n, size=n)  
        # default_rng(SEED).integers(low high, rand_cnt) 

        choice = allocate_vectorized(
            pred_scores[index],
            pred_costs[index],
            budget_multiplier=budget_multiplier,
            safety_ratio=safety_ratio,
            risk_multiplier=risk_multiplier,
        )

        real_light_total = real_costs[index, 0].sum()
        # gather realized cost/score for the chosen column, per resampled row

        chosen_real_cost = real_costs[index, choice]
        chosen_real_score = real_scores[index, choice]

        real_total = chosen_real_cost.sum()

        budget_limit = real_light_total * budget_multiplier

        if real_total > budget_limit:
            overruns += 1
        else:
            score_total += float(chosen_real_score.mean())

    return overruns / k, score_total / k
# (Risk 인자) × (Safety Ratio 인자)의 모든 조합에 대해 라우팅 선택 과정 가상으로 돌려보고
# 가장 점수가 높은 최선의 파라미터 조합 선택
# return overruns/k (n번중 몇번 예산 넘겼는지 확률), score_total/k(평균 품질 성능지표)



def calibrate_tier(
    pred_scores,
    pred_costs,
    real_scores,
    real_costs,
    *,
    tier: str,
    budget_multiplier: float,
    risk_high: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Returns (best_safety_ratio, overrun_probability, mean_score) for one
    (tier, risk_high) pair, searching the safety grid."""

    # * == * 뒤에 오는 인자는 무조건 키워드인자로 전달받음
    # tier=tier, risk_high=risk_high처럼 이름을 직접 붙여서 호출해야 함(강제)

    risk_multiplier = np.array([1.0, 1.0, risk_high])
    # think모델에 현재의 risk_high candidate 값을 적용하기 위한 temp multiplier

    best = None  # (mean_score, safety_ratio, overrun)
    fallback = None  # lowest-overrun candidate, in case nothing meets the target

    # fallback = 예산 초과 기준 만족하는 완벽한 설정값이 하나도 없을 때
    #            그나마 예산 초과율이 가장 낮았던 요소를 저장  

    for safety_ratio in SAFETY_GRID:
        # SAFETY_GRID == 안전 마진 후보군 list

        overrun_prob, mean_score = bootstrap_evaluate(
            pred_scores,
            pred_costs,
            real_scores,
            real_costs,
            budget_multiplier=budget_multiplier,
            safety_ratio=safety_ratio,
            risk_multiplier=risk_multiplier,
            rng=rng,
        )
        # return overruns/k (n번중 몇번 예산 넘겼는지 확률), 
        #        score_total/k(평균 품질 성능지표)


        if fallback is None or overrun_prob < fallback[2]:
            # fallback 비어있거나 예산초과율이 갱신되었으면 기록(하한)
            fallback = (mean_score, safety_ratio, overrun_prob)

        if overrun_prob <= OVERRUN_TARGET:
            # 예산 초과율 hyperparameter 이내에서만.
            if best is None or mean_score > best[0]:
                # score 가 가장 좋은 best candidate select
                best = (mean_score, safety_ratio, overrun_prob)
                

    chosen = best if best is not None else fallback
    mean_score, safety_ratio, overrun_prob = chosen
    return safety_ratio, overrun_prob, mean_score
# safety_ratio, overrun_prob(위험도), mean_score(점수) return




def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dev-input", type=Path, required=True)
    parser.add_argument("--dev-outcomes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    # dev data로 tier_safety_ratios, risk_multiplier 채워넣는 코드
    # dev data == validation data. artifact 파일에 추가할 2가지 인자 도출에만 사용

    policy = load_bundled_policy()
    artifact = load_artifact(args.artifact)
    dev_inputs = load_input(args.dev_input)
    dev_outcomes = load_outcomes(args.dev_outcomes)
    # type 한번씩 check. 다 써오던 인자들
    # train 때랑 동일하지만, dev (validation) data 가져오는 것

    print(f"Dev 문항 수: {len(dev_inputs.episodes)}")
    print("예측 계산 중...")

    pred_scores, pred_costs, real_scores, real_costs = build_matrices(
        dev_inputs, dev_outcomes, policy, artifact
    ) # in, out, policy, artifact =>  scores/costs 의 예측값, 실제값 return 

    rng = np.random.default_rng(RNG_SEED)
    # random number generator (default_rng = 개선된 rand()함수)

    weights = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
    budget_multipliers = {tier: float(policy.tiers[tier].budget_multiplier) for tier in TIERS}
    # budget_multiplier == 각 티어별 예산 가중치. (policy.v1schema, routing-policy.v1 ref.)
    # tier : multipliers  dictionary 구성

    print(f"\n부트스트랩 K={BOOTSTRAP_K}, 목표 초과율 <= {OVERRUN_TARGET:.0%}\n")
    # BOOTSTRAP_K = 300 (부트스트랩 시도 횟수. : 앙상블이랑 헷갈리지 X. 고전통계 부트스트랩)
    # OVERRRUN_TARGET = 0.1 (최대 허용예산 초과율 : 목표로 하는 최대예산초과확률 한도)
    # n번 시뮬레이션 했을 때 예산 초과할 확률이 p 이하가 되는 규제 피라미터(2개) 탐색하는 과정.
    

    # Dev를 복원추출로 300번 재표본 → 고정모델로 예측 → 배분결과 실제값으로 채점 → 예산초과확률 집계
    # 진짜 모집단(비공개 평가셋)에서 결과가 얼마나 흔들릴지를 근사하는 불확실성 추정 기법

    best_overall = None  
    # (weighted_score, risk_high, {tier: (safety, overrun, score)})

    for risk_high in RISK_HIGH_GRID: 
        # RISK_HIGH_GRID = think 모델 임계값 후보 list

        per_tier = {}
        weighted_score = 0.0
        for tier in TIERS:
            safety_ratio, overrun_prob, mean_score = calibrate_tier(
                pred_scores,
                pred_costs,
                real_scores,
                real_costs,
                tier=tier,
                budget_multiplier=budget_multipliers[tier],
                risk_high=risk_high,
                rng=rng,
            )
            per_tier[tier] = (safety_ratio, overrun_prob, mean_score)
            weighted_score += weights[tier] * mean_score


        print(
            f"risk[axk1-think]={risk_high:>4.2f}  가중점수={weighted_score:.6f}  "
            + "  ".join(
                f"{tier}: safety={per_tier[tier][0]:.2f} 초과율={per_tier[tier][1]:.3f} "
                f"점수={per_tier[tier][2]:.4f}"
                for tier in TIERS
            )
        )

        if best_overall is None or weighted_score > best_overall[0]:
            best_overall = (weighted_score, risk_high, per_tier)

    # risk_high :최외곽 최적화. 큰 틀에서 위험도계수 돌며 정함
    # tier : 서비스 등급당 티어별 예산 승수 (budget multiplier) 적용
    # safety_ratio : SAFETY_GRID 후보에서 bootstrap eval :
    # bootstrap에서 복원추출해서 문항 배정 (choicce)

    weighted_score, risk_high, per_tier = best_overall
    print(f"\n선택: risk[axk1-think]={risk_high}  가중점수(부트스트랩 평균)={weighted_score:.6f}")

    for tier in TIERS:
        safety_ratio, overrun_prob, mean_score = per_tier[tier]
        print(f"  {tier:9} safety_ratio={safety_ratio:.2f}  초과율={overrun_prob:.3f}  점수={mean_score:.4f}")
    # 티어별 요소 확인



    # Read the raw JSON (not the parsed dataclass -- _team_parse_artifact drops
    # dense_feature_names since inference doesn't need it) and only touch the
    # two fields this script is responsible for, so nothing else is lost.

    artifact_dict = json.loads(args.artifact.read_text(encoding="utf-8"))

    # tier_safety_ratios (등급별 하나씩, fast/balanced/premium)
    # 등급에 허용된 예산(budget_multiplier)을 실제로 몇 %까지만 쓸지"를 정하는 값

    # risk_multiplier (모델별 하나씩, 사실상 axk1-think만 1.0이 아닌 값)
    #  λ-이분탐색에서 "점수 대비 비용" 비교할때 이 모델의 예측 비용 얼마나 부풀려서 취급할지
    artifact_dict["tier_safety_ratios"] = {tier: per_tier[tier][0] for tier in TIERS}
    artifact_dict["risk_multiplier"] = {
        MODEL_IDS[0]: 1.0,
        MODEL_IDS[1]: 1.0,
        MODEL_IDS[2]: risk_high,
    }


    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact_dict, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nOK: {args.out}에 보정된 안전계수·위험계수를 저장했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
