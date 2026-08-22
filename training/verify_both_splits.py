# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Run the official self-check on BOTH public splits and print them together.

Exists because measuring only Dev hid a real failure: a version that scored
0.6773 on Dev was simultaneously blowing Premium's budget on Train (cost
ratio 4.13 > 4.0), which scores that tier 0 and would have dropped the real
total to 0.4510. Dev-only numbers looked like an improvement the whole time.

Every candidate gets checked here before it goes near main.

    py -3.13 training/verify_both_splits.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ossp_router.protocol import TIERS  # noqa: E402

SPLITS = {
    "dev": (
        REPO_ROOT / "data/materialized/dev/inputs.json",
        REPO_ROOT / "data/dev/outcomes.json",
    ),
    "train": (
        REPO_ROOT / "data/materialized/train/inputs.json",
        REPO_ROOT / "data/train/outcomes.json",
    ),
}


def run_split(name: str, inputs: Path, outcomes: Path, workdir: Path) -> dict:
    submissions = workdir / name
    submissions.mkdir(parents=True, exist_ok=True)
    env_python = sys.executable
    for tier in TIERS:
        subprocess.run(
            [
                env_python,
                str(REPO_ROOT / "container/entrypoint.py"),
                "--input", str(inputs),
                "--tier", tier,
                "--output", str(submissions / f"{tier}.json"),
            ],
            check=True,
            capture_output=True,
            env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")},
        )
    report = workdir / f"{name}-report.json"
    subprocess.run(
        [
            env_python, "-m", "ossp_router.cli", "self-check",
            "--input", str(inputs),
            "--outcomes", str(outcomes),
            "--submissions", str(submissions),
            "--report", str(report),
        ],
        check=True,
        capture_output=True,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    return json.loads(report.read_text(encoding="utf-8"))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        reports = {
            name: run_split(name, inputs, outcomes, workdir)
            for name, (inputs, outcomes) in SPLITS.items()
        }

    any_failed = False
    print(f"{'':10}{'dev':>12}{'train':>12}")
    print(f"{'final':10}" + "".join(f"{float(reports[s]['final_score']):>12.6f}" for s in SPLITS))
    for tier in TIERS:
        cells = ""
        for split in SPLITS:
            t = reports[split]["tiers"][tier]
            mark = "" if t["budget_passed"] else " !"
            if not t["budget_passed"]:
                any_failed = True
            cells += f"{float(t['tier_score']):>10.4f}{mark:<2}"
        print(f"{tier:10}{cells}")
    print("\n예산 비용비율:")
    for tier in TIERS:
        ratios = "  ".join(
            f"{split}={float(reports[split]['tiers'][tier]['budget_ratio']):.4f}"
            for split in SPLITS
        )
        limit = float(reports["dev"]["tiers"][tier]["budget_multiplier"])
        print(f"  {tier:9} 한도={limit:.2f}  {ratios}")

    # Score alone does not say whether a version is safe to ship: a higher
    # number bought by spending closer to the limit can be strictly worse once
    # the hidden split shifts. So report the margin in the unit that matters --
    # how many times the observed Dev->Train shift it can absorb. hash_regex
    # died at 0.07x (0.4% margin against a 5.4% shift); under ~2x deserves a
    # second look.
    print("\n안전 여유 (현재 사용량 대비):")
    for tier in TIERS:
        r = {sp: float(reports[sp]["tiers"][tier]["budget_ratio"]) for sp in SPLITS}
        limit = float(reports["dev"]["tiers"][tier]["budget_multiplier"])
        worst = max(r.values())
        margin = (limit - worst) / worst
        shift = abs(r["train"] - r["dev"]) / r["dev"]
        cover = margin / shift if shift > 1e-9 else float("inf")
        print(f"  {tier:9} 여유={margin:6.1%}  split간 변동={shift:5.1%}  -> 관측 변동의 {cover:5.1f}배 버팀")

    if any_failed:
        print("\n!! 한 split 이상에서 예산 초과 -- 해당 등급은 0점 처리된다.")
        return 1
    print("\n두 split 모두 예산 통과.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
