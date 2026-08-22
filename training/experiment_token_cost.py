# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Does splitting cost into computed-input + predicted-output help?

Every model prices output at exactly 4x input, so

    cost_m = input_rate_m * (input_tokens + 4 * output_tokens)

and input_tokens is 99.7% determined by prompt length (R^2 0.9970,
corr 0.9985 on Train). It is also 49.6% / 48.7% / 13.1% of each model's cost.
So today's single log-cost head spends capacity predicting a quantity we can
just compute, and carries its noise into the half of the cost that is free.

This measures the alternative offline -- fit the head on log(output_tokens),
add the analytic input term back at the end -- before touching heuristic.py.
Reported both as cost accuracy and, what actually matters, as achievable
score at a matched overrun bound.

    py -3.13 training/experiment_token_cost.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "training"))

from ossp_router.heuristic import (  # noqa: E402
    _TEAM_ARTIFACT_JSON,
    _team_raw_feature_vector,
    episode_text,
)
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)
from train_router import fit_ridge, oof_predictions, predict_ridge  # noqa: E402

CANDIDATES = (0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 10_000.0, 100_000.0)


def collect(split: str, hash_bins: int):
    batch = load_input(REPO_ROOT / f"data/materialized/{split}/inputs.json")
    index = {(o.episode_id, o.model_id): o
             for o in load_outcomes(REPO_ROOT / f"data/{split}/outcomes.json").outcomes}
    rows, chars, inp, out = [], [], [], []
    for episode in batch.episodes:
        if (episode.episode_id, MODEL_IDS[0]) not in index:
            continue
        rows.append(_team_raw_feature_vector(episode, hash_bins))
        chars.append(len(episode_text(episode)))
        inp.append([index[(episode.episode_id, m)].input_tokens for m in MODEL_IDS])
        out.append([index[(episode.episode_id, m)].output_tokens for m in MODEL_IDS])
    return (np.array(rows, dtype=float), np.array(chars, dtype=float),
            np.array(inp, dtype=float), np.array(out, dtype=float))


def main() -> int:
    import json
    artifact = json.loads(_TEAM_ARTIFACT_JSON)
    hash_bins = artifact["hash_bins"]
    policy = load_bundled_policy()
    unit = float(policy.token_unit)
    rate_in = np.array([float(policy.models[m].input_token_rate) for m in MODEL_IDS]) / unit
    rate_out = np.array([float(policy.models[m].output_token_rate) for m in MODEL_IDS]) / unit

    Xtr, ctr, itr, otr = collect("train", hash_bins)
    Xdv, cdv, idv, odv = collect("dev", hash_bins)
    real_tr = itr * rate_in + otr * rate_out
    real_dv = idv * rate_in + odv * rate_out

    # input_tokens from characters -- one least-squares line per model
    A_tr = np.vstack([ctr, np.ones_like(ctr)]).T
    A_dv = np.vstack([cdv, np.ones_like(cdv)]).T
    coef = np.linalg.lstsq(A_tr, itr, rcond=None)[0]
    in_tr, in_dv = A_tr @ coef, A_dv @ coef
    print("입력토큰 = a*문자수 + b  (모델별)")
    for j, m in enumerate(MODEL_IDS):
        err = np.abs(in_dv[:, j] - idv[:, j]).mean() / idv[:, j].mean()
        print(f"  {m:12} a={coef[0, j]:.5f} b={coef[1, j]:7.1f}   dev 평균 상대오차 {err:.2%}")

    # output head, on logs, alpha by out-of-fold MSE
    log_out = np.log(np.maximum(otr, 1.0))
    best = min(CANDIDATES,
               key=lambda a: float(((oof_predictions(Xtr, log_out, folds=5, alpha=a)
                                     - log_out) ** 2).mean()))
    mean, scale, intercept, coefficients = fit_ridge(Xtr, log_out, best)
    oof = oof_predictions(Xtr, log_out, folds=5, alpha=best)
    smear = np.exp(log_out).sum(axis=0) / np.exp(oof).sum(axis=0)
    print(f"\noutput head: alpha={best:g}  배치합 보정계수="
          + " ".join(f"{m}={s:.4f}" for m, s in zip(MODEL_IDS, smear)))

    out_dv = np.exp(predict_ridge(Xdv, mean, scale, intercept, coefficients)) * smear
    out_tr = np.exp(predict_ridge(Xtr, mean, scale, intercept, coefficients)) * smear
    new_dv = in_dv * rate_in + out_dv * rate_out
    new_tr = in_tr * rate_in + out_tr * rate_out

    cur = np.load(REPO_ROOT / "build/search-matrices.npz")
    print(f"\n{'':22}{'현재(log-cost)':>16}{'신규(입력계산)':>16}")
    for tag, new, real, key in (("dev", new_dv, real_dv, "dev"), ("train", new_tr, real_tr, "train")):
        old = cur[f"{key}__pred_costs"]
        for label, f in (("배치합 예측/실제", lambda p, r: p.sum(0) / r.sum(0)),
                         ("문항별 상대오차", lambda p, r: np.abs(p - r).sum(0) / r.sum(0))):
            o, n = f(old, real), f(new, real)
            for j, m in enumerate(MODEL_IDS):
                print(f"{tag+' '+label+' '+m:22}{o[j]:>16.4f}{n[j]:>16.4f}")

    # Level-correct on Train only, then ship the same factors to Dev. Doing it
    # per split would be fitting the thing we are trying to measure.
    fix = real_tr.sum(0) / new_tr.sum(0)
    print("\n배치합 레벨 보정 (Train에서만 산출): "
          + " ".join(f"{m}={f:.4f}" for m, f in zip(MODEL_IDS, fix)))
    new_dv, new_tr = new_dv * fix, new_tr * fix

    out_path = REPO_ROOT / "build/v18-matrices.npz"
    payload = {}
    for key, new, real in (("dev", new_dv, real_dv), ("train", new_tr, real_tr)):
        payload[f"{key}__pred_scores"] = cur[f"{key}__pred_scores"]
        payload[f"{key}__pred_costs"] = new
        payload[f"{key}__pred_rank_costs"] = new
        payload[f"{key}__real_scores"] = cur[f"{key}__real_scores"]
        payload[f"{key}__real_costs"] = real
    payload["budget_multipliers"] = cur["budget_multipliers"]
    np.savez_compressed(out_path, **payload)
    print(f"OK: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
