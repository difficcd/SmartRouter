# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0
"""시연 영상용 -- 컨테이너가 낸 제출 파일을 한 화면으로 요약한다."""
import collections
import json
import sys

path = sys.argv[1]
doc = json.load(open(path, encoding="utf-8"))
dec = doc["decisions"]
print(f"  challenge_id : {doc['challenge_id']}")
print(f"  policy_id    : {doc['policy_id']}")
print(f"  tier / split : {doc['tier']} / {doc['split']}")
print(f"  결정         : {len(dec)}건")
print()
counts = collections.Counter(d["model_id"] for d in dec)
COST = {"ax31-light": "기준선", "ax31": "2.1배", "axk1-think": "23.8배"}
for name, n in counts.most_common():
    print(f"    {name:12s} {n:4d}건  {n / len(dec) * 100:5.1f}%   (light 대비 {COST.get(name, '?')})")
print()
print(f"  첫 항목: {dec[0]}")
