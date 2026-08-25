# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Rebuild curve files from the blobs the Colab slices print.

Each slice ends by printing a gzip+base64 blob of its curve file between
BEGIN<slice> and END<slice> markers, so the result comes back without any file
transfer. This reverses that.

The header line each slice prints carries a row count and a SHA-256 prefix, and
both are checked here rather than trusted: a slice that quietly ran a different
grid, or a blob that got truncated in a copy-paste, would otherwise look like a
success and silently shrink the search.

    py -3.13 training/restore_colab_curves.py --tag v22 --input pasted.txt
    py -3.13 training/restore_colab_curves.py --tag v22 --input -   # stdin
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECT_ROWS = 24

BLOB = re.compile(r"BEGIN(s\d)\s*\n(.*?)\nEND\1", re.DOTALL)
HEADER = re.compile(
    r"SLICE=(s\d)\s+TIER=(\w+)\s+JOBS=(\d+)\s+PTS=(\d+)\s+SHA=([0-9a-f]{12})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="v22, v23, ...")
    parser.add_argument("--input", required=True, help="붙여넣은 텍스트 파일, 또는 - 로 stdin")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "build")
    args = parser.parse_args(argv)

    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(
        encoding="utf-8", errors="replace")

    # Output captured from the browser arrives as a JSON-encoded string, so its
    # line breaks are two-character escapes rather than newlines and the blob
    # markers never match. Unwrap that case; a hand-pasted file is unaffected.
    stripped = text.lstrip()
    if stripped.startswith('"'):
        try:
            text = json.loads(stripped)
        except ValueError:
            pass

    # Pair each blob with the header immediately preceding it. Keying by
    # slice id would let the last header in the page win, so a blob could be
    # checked against a header that does not belong to it -- which rejects the
    # genuine copy and admits the impostor, the exact inverse of the check.
    heads = [(m.start(), m.groups()) for m in HEADER.finditer(text)]

    def header_before(pos, slice_id):
        cands = [g for s, g in heads if s < pos and g[0] == slice_id]
        return cands[-1] if cands else None

    found = [(m.group(1), m.group(2), m.start()) for m in BLOB.finditer(text)]
    if not found:
        print("블롭을 찾지 못했습니다. BEGINs1 ... ENDs1 구간이 통째로 들어왔는지 확인하세요.")
        return 1

    # Count distinct slices, not blob occurrences. A captured page repeats the
    # same output under nested elements, so one slice can appear several times;
    # counting appearances would report 6/6 for five slices plus a duplicate,
    # and the search would quietly be missing a sixth of its grid.
    seen = {}
    for slice_id, payload, pos in found:
        raw = gzip.decompress(base64.b64decode("".join(payload.split())))
        digest = hashlib.sha256(raw).hexdigest()[:12]
        rows = json.loads(raw)["rows"]

        head = header_before(pos, slice_id)
        problems = []
        if len(rows) != EXPECT_ROWS:
            problems.append(f"조합 {len(rows)}개 (기대 {EXPECT_ROWS})")
        if head and head[4] != digest:
            problems.append(f"SHA 불일치 {digest} != {head[4]}")
        if head and int(head[2]) != len(rows):
            problems.append(f"헤더 JOBS={head[2]} != 실제 {len(rows)}")
        if problems:
            print(f"!! {slice_id}: " + ", ".join(problems))
            continue

        if slice_id in seen:
            if seen[slice_id] != digest:
                print(f"!! {slice_id}: 같은 조각이 서로 다른 내용으로 두 번 "
                      f"({seen[slice_id]} vs {digest}) -- 어느 쪽이 맞는지 알 수 없음")
                return 1
            continue

        path = args.out_dir / f"{args.tag}-curve-{slice_id}.json"
        path.write_bytes(raw)
        pts = sum(len(r["curve"]) for r in rows)
        print(f"OK {slice_id}: {len(rows)}조합 {pts}점 SHA={digest} -> {path.name}")
        seen[slice_id] = digest

    ok = len(seen)
    print()
    print(f"복원 {ok}/6  ({', '.join(sorted(seen))})")
    missing = sorted({f's{n}' for n in range(1, 7)} - set(seen))
    if missing:
        print(f"빠진 조각: {', '.join(missing)}")
    if ok != 6:
        print("6개가 모두 있어야 보정으로 넘어갑니다.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
