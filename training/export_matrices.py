# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Dump the per-episode prediction/outcome matrices the safety search needs.

The bootstrap search in calibrate_safety.py is by far the most expensive step
in the pipeline, and it never touches prompt text -- it only needs, per
episode and per model, four numbers: predicted score, predicted cost
(budget fit), predicted cost (ranking fit), real score, real cost.

Exporting just those makes the search portable: it can run anywhere numpy
runs, including a free cloud notebook, without shipping a single prompt off
this machine (which also keeps the AIME redistribution terms a non-issue).
The whole file is a few hundred KB.

    py -3.13 training/export_matrices.py --artifact training/artifact.json \
        --out build/search-matrices.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "training"))

from calibrate_safety import build_matrices, load_artifact  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, default=REPO_ROOT / "training/artifact.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "build/search-matrices.npz")
    args = parser.parse_args(argv)

    policy = load_bundled_policy()
    artifact = load_artifact(args.artifact)

    payload = {}
    for split, inputs_path, outcomes_path in [
        ("dev", "data/materialized/dev/inputs.json", "data/dev/outcomes.json"),
        ("train", "data/materialized/train/inputs.json", "data/train/outcomes.json"),
    ]:
        inputs = load_input(REPO_ROOT / inputs_path)
        outcomes = load_outcomes(REPO_ROOT / outcomes_path)
        names = ("pred_scores", "pred_costs", "pred_rank_costs", "real_scores", "real_costs")
        for name, matrix in zip(names, build_matrices(inputs, outcomes, policy, artifact)):
            payload[f"{split}__{name}"] = matrix
        print(f"{split}: {len(inputs.episodes)} 문항")

    payload["budget_multipliers"] = np.array(
        [float(policy.tiers[tier].budget_multiplier) for tier in TIERS], dtype=np.float64
    )
    payload["tiers"] = np.array(list(TIERS))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    size_kb = args.out.stat().st_size / 1024
    print(f"OK: {args.out} ({size_kb:.0f} KB) -- 프롬프트 텍스트는 포함되지 않음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
