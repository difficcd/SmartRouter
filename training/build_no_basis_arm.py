# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Train and export the arm we never actually built: a clean 266-feature model.

v20b does not ship the trigonometric basis -- it ships a 326-coefficient fit
evaluated on 266 features, which is neither the linear model nor the nonlinear
one but a misspecified hybrid. So "v20b vs v21" has been comparing a broken
linear-ish model against a correct nonlinear one, and the third cell of that
table was never filled in.

It matters because of what the stress scenarios say. v21 wins the bootstrap
bound (0.210% vs 0.364%) but loses badly on composition shift: on a
length-skewed batch premium runs at 3.2498 against v20b's 2.5478. The
trigonometric basis is a nonlinear transform of length-derived features, so a
batch whose length distribution moves pushes those features outside the range
they were fit on -- exactly where a nonlinear term goes wrong faster than a
linear one. v20b looked robust there partly BECAUSE the bug was discarding the
nonlinear terms.

A properly trained 266 model should keep that robustness without the
misspecification, and it already predicts better than what v20b ships
(Dev score MSE 0.16413 against 0.17216).

Nothing about the shipped router changes here. _TEAM_BASIS_FREQS is patched to
empty for this process only, which makes _team_expand_basis a pass-through, so
both the training matrix and the exported prediction matrices come out 266 wide
and self-consistent -- the parser's width check recomputes the same formula and
agrees.

    py -3.13 training/build_no_basis_arm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "training"))

from ossp_router import heuristic  # noqa: E402

# Empty frequencies -> _team_expand_basis appends nothing -> width 266.
heuristic._TEAM_BASIS_FREQS = ()

import export_matrices  # noqa: E402
import train_router  # noqa: E402

ARTIFACT = REPO_ROOT / "build/artifact-nobasis.json"
MATRICES = REPO_ROOT / "training/tmp-v21nb-matrices.npz"


def main() -> int:
    print("=== 266 특징 (삼각함수 없음) 학습 ===", flush=True)
    rc = train_router.main([
        "--train-input", str(REPO_ROOT / "data/materialized/train/inputs.json"),
        "--train-outcomes", str(REPO_ROOT / "data/train/outcomes.json"),
        "--cost-fit", "classic",          # what main ships; changing two things
        "--out", str(ARTIFACT),           # at once is how a day got wasted before
    ])
    if rc != 0:
        return rc

    import json
    width = len(json.loads(ARTIFACT.read_text(encoding="utf-8"))["feature_mean"])
    print(f"\n특징 폭: {width}", flush=True)
    if width != 266:
        print("!! 266이 아니다 -- 패치가 듣지 않았다. 중단.", flush=True)
        return 1

    print("\n=== 행렬 추출 ===", flush=True)
    return export_matrices.main(["--artifact", str(ARTIFACT), "--out", str(MATRICES)])


if __name__ == "__main__":
    raise SystemExit(main())
