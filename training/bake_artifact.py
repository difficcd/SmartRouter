# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the embedded artifact JSON inside src/ossp_router/heuristic.py.

Why the artifact lives inside a .py file instead of its own resource file:
container/.dockerignore is a frozen allowlist (tests/test_repository_policy.py
asserts its exact byte content) that only ships specific files under
src/ossp_router/ and baselines/ into the image. There is no slot for a new
resource file, so the trained weights are embedded as text between two
marker comments in heuristic.py instead. This script is what keeps that
embedded copy in sync with training/artifact.json -- never hand-edit the
text between the markers.

Run this as the last step of the pipeline, after train_router.py and
calibrate_safety.py:

    py -3.13 training/train_router.py       --out training/artifact.json
    py -3.13 training/calibrate_safety.py    --artifact training/artifact.json --out training/artifact.json
    py -3.13 training/bake_artifact.py       --artifact training/artifact.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
HEURISTIC_PATH = REPO_ROOT / "src/ossp_router/heuristic.py"
BEGIN_MARKER = "# BEGIN GENERATED ARTIFACT\n"
END_MARKER = "# END GENERATED ARTIFACT\n"


def bake(artifact_json_path: Path, heuristic_path: Path = HEURISTIC_PATH) -> None:
    artifact_text = artifact_json_path.read_text(encoding="utf-8").strip()
    if "'''" in artifact_text:
        # Would break out of the raw triple-quoted string below. Never actually
        # happens for a JSON blob of numbers/short id strings, but check anyway.
        raise ValueError("artifact JSON에 ''' 문자열이 있어 그대로 임베드할 수 없습니다.")

    heuristic_text = heuristic_path.read_text(encoding="utf-8")
    begin = heuristic_text.find(BEGIN_MARKER)
    end = heuristic_text.find(END_MARKER)
    if begin == -1 or end == -1 or end < begin:
        raise ValueError(
            f"{heuristic_path}에서 BEGIN/END GENERATED ARTIFACT 마커를 못 찾았습니다."
        )
    begin_of_content = begin + len(BEGIN_MARKER)

    new_block = f"_TEAM_ARTIFACT_JSON = r'''\n{artifact_text}\n'''\n"
    new_text = heuristic_text[:begin_of_content] + new_block + heuristic_text[end:]
    heuristic_path.write_text(new_text, encoding="utf-8")
    print(f"OK: {artifact_json_path}의 내용을 {heuristic_path}에 구웠습니다.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, default=REPO_ROOT / "training/artifact.json")
    parser.add_argument("--heuristic", type=Path, default=HEURISTIC_PATH)
    args = parser.parse_args(argv)
    try:
        bake(args.artifact, args.heuristic)
    except (OSError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
