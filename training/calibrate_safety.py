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


def predict_batch(episodes, artifact):
    scores, costs = [], []
    for episode in episodes:
        episode_scores, episode_costs = _team_predict_episode(episode, artifact)
        scores.append(episode_scores)
        costs.append(episode_costs)
    return scores, costs


RNG_SEED = 20260815
BOOTSTRAP_K = 300
OVERRUN_TARGET = 0.01
SAFETY_GRID = [
    0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
    # budget_multiplier * safety_ratio < 1.0 always collapses to all-light
    # (max(1.0, ...) floor in allocate_vectorized's cap), so 0.80-1.00 is
    # the only region where a tier can actually promote anything -- worth
    # a finer step there instead of jumping straight to 0.90/1.00.
    0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00,
]
RISK_HIGH_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
RISK_MID_GRID = [1.0, 1.2, 1.5, 2.0]


def _real_cost(outcome, policy: RoutingPolicy) -> float:
    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    return float(
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )


def build_matrices(
    inputs: InputBatch, outcomes: OutcomeBatch, policy: RoutingPolicy, artifact
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (predicted_scores, predicted_costs, real_scores, real_costs), each (n, 3)."""
    predicted_scores, predicted_costs = predict_batch(inputs.episodes, artifact)
    outcome_index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}

    n = len(inputs.episodes)
    n_models = len(MODEL_IDS)
    pred_scores = np.empty((n, n_models), dtype=np.float64)
    pred_costs = np.empty((n, n_models), dtype=np.float64)
    real_scores = np.empty((n, n_models), dtype=np.float64)
    real_costs = np.empty((n, n_models), dtype=np.float64)
    for row, episode in enumerate(inputs.episodes):
        for col, model_id in enumerate(MODEL_IDS):
            pred_scores[row, col] = predicted_scores[row][model_id]
            pred_costs[row, col] = predicted_costs[row][model_id]
            outcome = outcome_index[(episode.episode_id, model_id)]
            real_scores[row, col] = float(outcome.score)
            real_costs[row, col] = _real_cost(outcome, policy)
    return pred_scores, pred_costs, real_scores, real_costs


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

    def choose(penalty: float) -> Tuple[np.ndarray, float]:
        ev = pred_scores - penalty * risk_multiplier[None, :] * pred_costs / light_total
        choice = np.argmax(ev, axis=1)  # ties -> first (cheapest) column, matches -index tie-break
        total = pred_costs[np.arange(len(choice)), choice].sum()
        return choice, total

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

    return choice


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
        index = rng.integers(0, n, size=n)
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


def calibrate_tier(
    pred_scores,
    pred_costs,
    real_scores,
    real_costs,
    *,
    tier: str,
    budget_multiplier: float,
    risk_mid: float,
    risk_high: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Returns (best_safety_ratio, overrun_probability, mean_score) for one
    (tier, risk_mid, risk_high) triple, searching the safety grid."""

    risk_multiplier = np.array([1.0, risk_mid, risk_high])
    best = None  # (mean_score, safety_ratio, overrun)
    fallback = None  # lowest-overrun candidate, in case nothing meets the target
    for safety_ratio in SAFETY_GRID:
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
        if fallback is None or overrun_prob < fallback[2]:
            fallback = (mean_score, safety_ratio, overrun_prob)
        if overrun_prob <= OVERRUN_TARGET:
            if best is None or mean_score > best[0]:
                best = (mean_score, safety_ratio, overrun_prob)
    chosen = best if best is not None else fallback
    mean_score, safety_ratio, overrun_prob = chosen
    return safety_ratio, overrun_prob, mean_score


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dev-input", type=Path, required=True)
    parser.add_argument("--dev-outcomes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    policy = load_bundled_policy()
    artifact = load_artifact(args.artifact)
    dev_inputs = load_input(args.dev_input)
    dev_outcomes = load_outcomes(args.dev_outcomes)

    print(f"Dev 문항 수: {len(dev_inputs.episodes)}")
    print("예측 계산 중...")
    pred_scores, pred_costs, real_scores, real_costs = build_matrices(
        dev_inputs, dev_outcomes, policy, artifact
    )

    rng = np.random.default_rng(RNG_SEED)
    budget_multipliers = {tier: float(policy.tiers[tier].budget_multiplier) for tier in TIERS}

    # risk_multiplier used to be shared across all three tiers (one risk_mid,
    # risk_high picked to maximize the tiers' WEIGHTED score together). Real
    # Dev self-check showed fast and premium want opposite risk_high values
    # (fast: trust axk1's now-corrected cost estimate fully so it isn't
    # priced out of its already-razor-thin 1.25x budget; premium: stay
    # cautious on axk1's cost since it has the most room to absorb a miss)
    # -- a shared value forces a compromise that helps neither. Each tier
    # now searches its own (risk_mid, risk_high, safety_ratio) independently
    # and keeps whichever wins for itself; risk_multiplier is per-tier below.
    print(f"\n부트스트랩 K={BOOTSTRAP_K}, 목표 초과율 <= {OVERRUN_TARGET:.0%}  (등급별 독립 탐색)\n")
    per_tier = {}  # tier -> (risk_mid, risk_high, safety_ratio, overrun, score)
    for tier in TIERS:
        best = None
        for risk_high in RISK_HIGH_GRID:
            for risk_mid in RISK_MID_GRID:
                safety_ratio, overrun_prob, mean_score = calibrate_tier(
                    pred_scores,
                    pred_costs,
                    real_scores,
                    real_costs,
                    tier=tier,
                    budget_multiplier=budget_multipliers[tier],
                    risk_mid=risk_mid,
                    risk_high=risk_high,
                    rng=rng,
                )
                if best is None or mean_score > best[4]:
                    best = (risk_mid, risk_high, safety_ratio, overrun_prob, mean_score)
        per_tier[tier] = best
        risk_mid, risk_high, safety_ratio, overrun_prob, mean_score = best
        print(
            f"{tier:9} risk[ax31]={risk_mid:.2f} risk[axk1-think]={risk_high:.2f} "
            f"safety={safety_ratio:.2f}  초과율={overrun_prob:.3f}  점수={mean_score:.4f}"
        )

    # Read the raw JSON (not the parsed dataclass -- _team_parse_artifact drops
    # dense_feature_names since inference doesn't need it) and only touch the
    # two fields this script is responsible for, so nothing else is lost.
    artifact_dict = json.loads(args.artifact.read_text(encoding="utf-8"))
    artifact_dict["tier_safety_ratios"] = {tier: per_tier[tier][2] for tier in TIERS}
    artifact_dict["risk_multiplier"] = {
        tier: {
            MODEL_IDS[0]: 1.0,
            MODEL_IDS[1]: per_tier[tier][0],
            MODEL_IDS[2]: per_tier[tier][1],
        }
        for tier in TIERS
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
