# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0
#
# 원본: src/ossp_router/heuristic.py, 227~2750행 (SKT 원본 파일 맨 아래 이어붙인 섹션 전체)
#
# 이 사본은 밑에서 쓰는 이름(re, Episode, episode_text, _parser 등)이 전부
# 원본 1~226행(SKT 베이스라인 부분)에서 오는 거라, IDE가 "정의 안 된 이름"으로
# 노란 줄을 긋는 걸 막으려고 그 최소 의존성만 앞에 그대로 옮겨왔습니다.
# (진짜 저장소 파일은 이 사본과 무관하게 그대로입니다.)
#
# 366~2571행(_TEAM_ARTIFACT_JSON = r'''...''', 학습된 계수 JSON 2,205줄)은
# 순수 데이터라서 생략 -- 구조는 training/artifact.json 참고. 대신 IDE가
# 이름을 못 찾아 노란 줄 긋는 걸 막으려고 빈 자리표시자 하나만 넣어둠.

# ============================================================================
# 원본 1~226행에서 가져온 최소 의존성 (실제로는 SKT 베이스라인 코드 안에 있음)
# ============================================================================

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from .protocol import (
    TIERS,
    Decision,
    Episode,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    dumps_json,
    load_bundled_policy,
    load_input,
    load_policy,
    parse_submission,
    submission_to_dict,
)

_CODE_MARKERS = re.compile(
    r"```|(?:^|\s)(?:def|class|function|SELECT|FROM|import|#include)\b|"
    r"[{};]\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MATH_MARKERS = re.compile(r"[=+\-*/^∑∫√≈≠≤≥<>]|\\(?:frac|sum|int|sqrt)\b")
_SENTENCE_END = re.compile(r"[.!?。！？]")
_REASONING_WORDS = re.compile(
    r"\b(?:prove|derive|reason|analyze|explain why|algorithm|complexity|"
    r"증명|유도|추론|분석|알고리즘|복잡도)\b",
    re.IGNORECASE,
)


def episode_text(episode: Episode) -> str:
    """Return only the prompt or message content available at routing time."""

    if episode.prompt is not None:
        return episode.prompt
    assert episode.messages is not None
    return "\n".join(message.content for message in episode.messages)


def write_submission_atomic(path: Path, submission: Submission) -> None:
    """Write one result without leaving a valid-looking partial JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            dumps_json(submission_to_dict(submission)),
            encoding="utf-8",
        )
        temporary.chmod(0o644)
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-run",
        description="프롬프트 기반 baseline 라우터를 한 등급에 대해 실행합니다.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    return parser


# ============================================================================
# heuristic.py 227~365행 (여기부터 진짜 "difficcd team router additions" 섹션)
# ============================================================================

# BEGIN difficcd team router additions
# ============================================================================
#
# Everything below this line is added on top of SKT's reference baseline
# above (untouched -- select_model()/make_submission()/main() above still
# work exactly as before, so baselines/prompt_heuristic.py and the tests
# that import from this module are unaffected).
#
# Why appended here instead of a separate package: container/.dockerignore
# is a frozen allowlist (tests/test_repository_policy.py asserts its exact
# content) that only ships specific files under src/ossp_router/ and
# baselines/. A new top-level package would silently never reach the image.
# router-run's contract (docs/RUNTIME.md) doesn't care which file provides
# it, so this file -- explicitly named as the one to replace in
# baselines/README.md -- is where it goes.

import math as _team_math  # noqa: E402
from typing import Mapping as _TeamMapping  # noqa: E402
from typing import Tuple as _TeamTuple  # noqa: E402

from .protocol import MODEL_IDS as _TEAM_MODEL_IDS  # noqa: E402
from .protocol import policy_sha256 as _team_policy_sha256  # noqa: E402
from .protocol import loads_json as _team_loads_json  # noqa: E402


# ---- features: prompt text -> fixed-length numeric vector ----------------
# Dense hand-picked counts/ratios + hashed word unigram/bigram counts
# (FNV-1a, signed -- no vocabulary/dictionary is stored, so no prompt text
# ever ends up in the artifact below).

_TEAM_FNV_OFFSET_BASIS = 0xCBF29CE484222325
_TEAM_FNV_PRIME = 0x100000001B3
_TEAM_MASK_64 = (1 << 64) - 1

_TEAM_WORD = re.compile(r"[0-9A-Za-z가-힣]+")
_TEAM_DENSE_FEATURE_NAMES: _TeamTuple[str, ...] = (
    "character_count_log1p",
    "word_count_log1p",
    "sentence_count_log1p",
    "message_count",
    "hangul_ratio",
    "code_marker_count",
    "math_marker_count",
    "numeric_density",
    "long_context",
    "reasoning_marker_count",
)


def _team_fnv1a64(data: bytes) -> int:
    value = _TEAM_FNV_OFFSET_BASIS
    for byte in data:
        value ^= byte
        value = (value * _TEAM_FNV_PRIME) & _TEAM_MASK_64
    return value


def _team_hash_slot(token: str, hash_bins: int) -> _TeamTuple[int, float]:
    digest = _team_fnv1a64(token.encode("utf-8"))
    index = digest % hash_bins
    sign = 1.0 if (digest >> 63) & 1 else -1.0
    return index, sign


def _team_tokens(text: str) -> _TeamTuple[str, ...]:
    return tuple(match.group(0).lower() for match in _TEAM_WORD.finditer(text))


def _team_bigrams(tokens: Sequence[str]) -> _TeamTuple[str, ...]:
    return tuple(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))


def _team_dense_features(episode: Episode, text: str) -> _TeamTuple[float, ...]:
    characters = len(text)
    nonspace = sum(not ch.isspace() for ch in text) or 1
    hangul = sum("가" <= ch <= "힣" for ch in text)
    numbers = sum(ch.isdigit() for ch in text)
    words = _team_tokens(text)
    message_count = 1.0 if episode.prompt is not None else float(len(episode.messages or ()))
    sentence_count = max(1, len(_SENTENCE_END.findall(text)))
    return (
        _team_math.log1p(characters),
        _team_math.log1p(len(words)),
        _team_math.log1p(sentence_count),
        message_count,
        hangul / nonspace,
        float(len(_CODE_MARKERS.findall(text))),
        float(len(_MATH_MARKERS.findall(text))),
        numbers / nonspace,
        1.0 if characters >= 8_000 else 0.0,
        float(len(_REASONING_WORDS.findall(text))),
    )


def _team_raw_feature_vector(episode: Episode, hash_bins: int) -> _TeamTuple[float, ...]:
    text = episode_text(episode)
    vector = list(_team_dense_features(episode, text))
    vector.extend(0.0 for _ in range(hash_bins))
    offset = len(_TEAM_DENSE_FEATURE_NAMES)
    tokens = _team_tokens(text)
    for gram in (*tokens, *_team_bigrams(tokens)):
        index, sign = _team_hash_slot(gram, hash_bins)
        vector[offset + index] += sign
    return tuple(vector)


# ---- artifact: the trained model, frozen into one JSON blob --------------
# Embedded as text (not a separate resource file) because container/
# .dockerignore is a frozen allowlist that doesn't have a slot for one --
# see the module docstring above this section.


class _TeamLinearHead:
    __slots__ = ("intercept", "coefficients")

    def __init__(self, intercept: float, coefficients: _TeamTuple[float, ...]) -> None:
        self.intercept = intercept
        self.coefficients = coefficients


class _TeamArtifact:
    __slots__ = (
        "hash_bins",
        "policy_id",
        "policy_sha256",
        "feature_mean",
        "feature_scale",
        "score_heads",
        "log_cost_heads",
        "tier_safety_ratios",
        "risk_multiplier",
    )

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


# Regenerated by training/bake_artifact.py -- do not hand-edit the text below.


# ============================================================================
# [ 여기 366~2571행: _TEAM_ARTIFACT_JSON = r'''...''' 학습된 계수 JSON 삽입 자리 ]
# 2,205줄짜리 순수 데이터라 생략함. 안에 뭐가 들었는지는 training/artifact.json
# 파일을 직접 열어보거나, 대화에서 정리한 표 참고:
#   dense_feature_names(10) / feature_mean·scale(266개씩) /
#   score_heads·log_cost_heads(모델3 x {intercept, coefficients[266]}) /
#   policy_id / policy_sha256 / tier_safety_ratios / risk_multiplier
# ============================================================================

_TEAM_ARTIFACT_JSON = "{}"  # 자리표시자 -- 진짜 값은 training/artifact.json


# ============================================================================
# heuristic.py 2572~2750행
# ============================================================================

def _team_parse_artifact(text: str) -> _TeamArtifact:
    value = _team_loads_json(text)

    def head(raw) -> _TeamLinearHead:
        return _TeamLinearHead(
            intercept=float(raw["intercept"]),
            coefficients=tuple(float(c) for c in raw["coefficients"]),
        )

    return _TeamArtifact(
        hash_bins=int(value["hash_bins"]),
        policy_id=str(value["policy_id"]),
        policy_sha256=str(value["policy_sha256"]),
        feature_mean=tuple(float(v) for v in value["feature_mean"]),
        feature_scale=tuple(float(v) for v in value["feature_scale"]),
        score_heads={m: head(value["score_heads"][m]) for m in _TEAM_MODEL_IDS},
        log_cost_heads={m: head(value["log_cost_heads"][m]) for m in _TEAM_MODEL_IDS},
        tier_safety_ratios={t: float(value["tier_safety_ratios"][t]) for t in TIERS},
        risk_multiplier={m: float(value["risk_multiplier"][m]) for m in _TEAM_MODEL_IDS},
    )


_team_artifact_singleton: Optional[_TeamArtifact] = None


def _team_load_artifact() -> _TeamArtifact:
    global _team_artifact_singleton
    if _team_artifact_singleton is None:
        _team_artifact_singleton = _team_parse_artifact(_TEAM_ARTIFACT_JSON)
    return _team_artifact_singleton


# ---- predict: q̂(x, m) and ĉ(x, m) for one episode -----------------------


def _team_linear(head: _TeamLinearHead, values: Sequence[float]) -> float:
    return head.intercept + _team_math.fsum(c * v for c, v in zip(head.coefficients, values))


def _team_predict_episode(
    episode: Episode, artifact: _TeamArtifact
) -> _TeamTuple[_TeamMapping[str, float], _TeamMapping[str, float]]:
    raw = _team_raw_feature_vector(episode, artifact.hash_bins)
    standardized = tuple(
        (v - m) / s for v, m, s in zip(raw, artifact.feature_mean, artifact.feature_scale)
    )
    scores = {
        m: min(1.0, max(0.0, _team_linear(artifact.score_heads[m], standardized)))
        for m in _TEAM_MODEL_IDS
    }
    costs = {
        m: _team_math.exp(
            min(50.0, max(-50.0, _team_linear(artifact.log_cost_heads[m], standardized)))
        )
        for m in _TEAM_MODEL_IDS
    }
    light, mid, high = _TEAM_MODEL_IDS
    costs[mid] = max(costs[mid], costs[light] * (1.0 + 1e-12))
    costs[high] = max(costs[high], costs[mid] * (1.0 + 1e-12))
    return scores, costs


# ---- allocate: batch-level λ-bisection with per-model risk multiplier ------


def _team_select_models(
    predicted_scores: Sequence[_TeamMapping[str, float]],
    predicted_costs: Sequence[_TeamMapping[str, float]],
    *,
    budget_multiplier: float,
    safety_ratio: float,
    risk_multiplier: _TeamMapping[str, float],
) -> _TeamTuple[str, ...]:
    light_total = _team_math.fsum(row[_TEAM_MODEL_IDS[0]] for row in predicted_costs)
    cap = light_total * max(1.0, budget_multiplier * safety_ratio)

    def choose(penalty: float) -> _TeamTuple[_TeamTuple[str, ...], float]:
        selected = []
        for scores, costs in zip(predicted_scores, predicted_costs):
            model_id = max(
                _TEAM_MODEL_IDS,
                key=lambda candidate: (
                    scores[candidate]
                    - penalty * risk_multiplier[candidate] * costs[candidate] / light_total,
                    -_TEAM_MODEL_IDS.index(candidate),
                ),
            )
            selected.append(model_id)
        total = _team_math.fsum(
            costs[model_id] for costs, model_id in zip(predicted_costs, selected)
        )
        return tuple(selected), total

    selected, total = choose(0.0)
    if total > cap:
        low, high = 0.0, 1.0
        selected, total = choose(high)
        while total > cap and high < 2.0**40:
            low, high = high, high * 2.0
            selected, total = choose(high)
        for _iteration in range(40):
            middle = (low + high) / 2.0
            candidate_selected, candidate_total = choose(middle)
            if candidate_total <= cap:
                high = middle
                selected, total = candidate_selected, candidate_total
            else:
                low = middle
    if total > cap:
        selected = tuple(_TEAM_MODEL_IDS[0] for _row in predicted_scores)

    return selected


# ---- router_main: the real entry point (container/entrypoint.py uses this) -


def team_router_make_submission(
    inputs: InputBatch, policy: RoutingPolicy, tier: str
) -> Submission:
    artifact = _team_load_artifact()
    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("입력과 정책의 schema_version이 일치하지 않습니다.")
    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    if artifact.policy_id != policy.policy_id:
        raise ProtocolError("artifact와 정책의 policy_id가 다릅니다.")
    if artifact.policy_sha256 != _team_policy_sha256(policy):
        raise ProtocolError("artifact와 현재 정책의 SHA-256이 다릅니다 -- 재학습 필요.")

    predicted_scores = []
    predicted_costs = []
    for episode in inputs.episodes:
        scores, costs = _team_predict_episode(episode, artifact)
        predicted_scores.append(scores)
        predicted_costs.append(costs)

    selected = _team_select_models(
        predicted_scores,
        predicted_costs,
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=artifact.tier_safety_ratios[tier],
        risk_multiplier=artifact.risk_multiplier,
    )

    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, model_id)
            for episode, model_id in zip(inputs.episodes, selected)
        ),
    )
    return parse_submission(submission_to_dict(submission))


def router_main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = (
            load_policy(args.policy) if args.policy is not None else load_bundled_policy()
        )
        submission = team_router_make_submission(inputs, policy, args.tier)
        write_submission_atomic(args.output, submission)
    except (OSError, ProtocolError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {args.tier} 제출 파일을 생성했습니다.")
    return 0


# ============================================================================
# END difficcd team router additions
