# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Mine per-model marker words from Train, with a split-half guard.

v8 built one word list by ranking words on their mean (axk1 score - light
score) over Train. That list feeds a single count feature, so every word it
contains is summed into one number -- which means a word favouring light and a
word favouring axk1 cancel each other out. This mines a separate list per
model instead, so the counts can point in different directions.

The selection guard matters more than the ranking. Picking the top words out
of thousands of noisy per-word means is the same winner's-curse machinery that
made v13's safety numbers look better than they were, so a word only qualifies
if it shows the same sign in BOTH halves of Train, split by episode index.
Words that only look good in one half are exactly the ones that will not
survive to the real test set.

    py -3.13 training/mine_marker_words.py --min-count 15 --top 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ossp_router.heuristic import _team_tokens, episode_text  # noqa: E402
from ossp_router.protocol import MODEL_IDS, load_input, load_outcomes  # noqa: E402

# Words that are obviously about the answer rather than the question. v8 threw
# these out by hand after seeing AIME answer values and stray topic nouns top
# the list; encoding the rule keeps the next pass honest.
_PURE_NUMBER = re.compile(r"^\d+$")


def load(inputs: Path, outcomes: Path):
    batch = load_input(inputs)
    index = {(o.episode_id, o.model_id): o for o in load_outcomes(outcomes).outcomes}
    rows = []
    for episode in batch.episodes:
        try:
            scores = {m: float(index[(episode.episode_id, m)].score) for m in MODEL_IDS}
        except KeyError:
            continue
        rows.append((set(_team_tokens(episode_text(episode))), scores))
    return rows


def advantages(rows, model, baseline, indices):
    """Mean (model - baseline) score per word, plus the sum of squares.

    The squares are what makes a significance test possible. Sign agreement
    across two halves is far too weak a filter on its own: a word with no real
    effect passes it about a quarter of the time, so out of ~2000 candidates
    roughly 500 sail through on luck alone. That is the same winner's-curse
    problem the v16 safety work ran into, and it needs the same answer -- test
    the effect, do not just rank it.
    """
    totals = defaultdict(float)
    squares = defaultdict(float)
    counts = defaultdict(int)
    for i in indices:
        words, scores = rows[i]
        gain = scores.get(model, 0.0) - scores.get(baseline, 0.0)
        for word in words:
            totals[word] += gain
            squares[word] += gain * gain
            counts[word] += 1
    return totals, squares, counts


def t_statistic(total, square, count):
    """mean / standard error, 0 when it cannot be computed."""
    if count < 2:
        return 0.0
    mean = total / count
    var = max(square / count - mean * mean, 0.0) * count / (count - 1)
    if var <= 0.0:
        return 0.0
    return mean / (var / count) ** 0.5


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path,
                        default=REPO_ROOT / "data/materialized/train/inputs.json")
    parser.add_argument("--outcomes", type=Path,
                        default=REPO_ROOT / "data/train/outcomes.json")
    parser.add_argument("--min-count", type=int, default=15,
                        help="각 절반에서 이 횟수 이상 등장해야 후보 (기본 %(default)s)")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--max-df", type=float, default=0.25,
        help="이 비율보다 자주 등장하는 단어는 제외 (기본 %(default)s). t통계량은 "
             "효과 크기가 아니라 빈도를 보상하므로, 상한을 두지 않으면 불용어가 "
             "상위를 차지하고 카운트가 사실상 '단어 수'가 되어 릿지가 기존 길이 "
             "특징과 중복으로 보고 계수를 0으로 눌러버린다 (v17 1차 시도 실패 원인).",
    )
    parser.add_argument(
        "--min-t", type=float, default=3.5,
        help="전체 Train 기준 t통계량 하한 (기본 %(default)s). 후보가 약 2000개라 "
             "Bonferroni로 0.05/2000 => z~3.9에 해당하는 수준을 잡은 것.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = load(args.inputs, args.outcomes)
    print(f"Train {len(rows)}문항, 절반씩 나눠 교차 확인 (min-count={args.min_count}/절반)\n")
    first = range(0, len(rows), 2)
    second = range(1, len(rows), 2)

    light = MODEL_IDS[0]
    whole = range(len(rows))
    selected = {}
    for model in MODEL_IDS[1:]:
        ta, sa, ca = advantages(rows, model, light, first)
        tb, sb, cb = advantages(rows, model, light, second)
        tw, sw, cw = advantages(rows, model, light, whole)
        shared = [w for w in ca
                  if ca[w] >= args.min_count and cb.get(w, 0) >= args.min_count
                  and not _PURE_NUMBER.match(w)]
        agree = [w for w in shared if ta[w] > 0 and tb[w] > 0]
        df_cap = args.max_df * len(rows)
        common = [w for w in agree if cw[w] > df_cap]
        scored = []
        for w in agree:
            if cw[w] > df_cap:
                continue
            t = t_statistic(tw[w], sw[w], cw[w])
            if t >= args.min_t:
                # rank on effect size, use significance only as a gate
                scored.append((tw[w] / cw[w], t, cw[w], w))
        scored.sort(reverse=True)
        picked = scored[: args.top]
        selected[model] = sorted(w for *_, w in picked)
        print(f"=== {model} 우위 단어 "
              f"(후보 {len(shared)} -> 부호 일치 {len(agree)} -> 빈출 제외 {len(common)}개 "
              f"-> t>={args.min_t} {len(scored)} -> 채택 {len(picked)})")
        print(f"{'단어':<14}{'평균우위':>10}{'t':>8}{'등장':>7}")
        for mean, t, n, w in picked[:15]:
            print(f"{w:<14}{mean:>10.4f}{t:>8.2f}{n:>7}")
        print(f"   전체: {' '.join(selected[model])}\n")

    if args.out:
        args.out.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"OK: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
