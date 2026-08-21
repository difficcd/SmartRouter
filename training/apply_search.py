# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Merge per-tier search results into the artifact.

search_standalone.py can run each tier anywhere (see its docstring); this
collects those JSONs and writes the chosen safety parameters into
training/artifact.json, exactly as calibrate_safety.py would have if it had
run all three tiers in one process.

    py -3.13 training/apply_search.py --artifact training/artifact.json \
        --results build/search-fast.json build/search-premium.json \
                  build/search-balanced.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ossp_router.protocol import MODEL_IDS, TIERS  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, default=REPO_ROOT / "training/artifact.json")
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    # A tier may arrive in several pieces when its risk_high grid was split
    # across machines (search_standalone.py --risk-high). Keep the best-scoring
    # entry per tier; identical tiers from a full search simply agree.
    merged = {}
    for path in args.results:
        chunk = json.loads(path.read_text(encoding="utf-8"))
        for tier, entry in chunk.items():
            if tier not in merged or entry["score"] > merged[tier]["score"]:
                merged[tier] = entry

    missing = [tier for tier in TIERS if tier not in merged]
    if missing:
        raise SystemExit(f"결과가 없는 등급: {missing}")

    for tier in TIERS:
        entry = merged[tier]
        print(
            f"{tier:9} risk[ax31]={entry['risk_mid']:.2f} "
            f"risk[axk1]={entry['risk_high']:.2f} safety={entry['safety_ratio']:.2f} "
            f"high_cap={entry['high_cap_ratio']:.2f} "
            f"share={entry.get('share_ratio', 1.0):.2f} "
            f"최악초과율={entry['overrun']:.3f} 점수={entry['score']:.4f}"
        )

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    artifact["tier_safety_ratios"] = {t: merged[t]["safety_ratio"] for t in TIERS}
    artifact["risk_multiplier"] = {
        t: {
            MODEL_IDS[0]: 1.0,
            MODEL_IDS[1]: merged[t]["risk_mid"],
            MODEL_IDS[2]: merged[t]["risk_high"],
        }
        for t in TIERS
    }
    artifact["high_cap_ratio"] = {t: merged[t]["high_cap_ratio"] for t in TIERS}
    artifact["share_ratio"] = {t: merged[t].get("share_ratio", 1.0) for t in TIERS}

    out = args.out or args.artifact
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nOK: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
