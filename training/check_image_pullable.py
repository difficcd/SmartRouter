# SPDX-FileCopyrightText: Copyright 2026 difficcd
# SPDX-License-Identifier: Apache-2.0

"""Ask a registry, anonymously, whether the submitted image can actually be got.

tools/validate_technical_submission.py checks the JSON's shape. It cannot check
that the digest names something a stranger can pull, and those are different
questions: a reference missing its registry prefix validates fine and then
resolves to a repository that does not exist, and a ghcr package left private
validates fine and then refuses anyone who is not its owner. Either way the
evaluator gets nothing and the submission scores nothing.

So this asks the registry directly, with no credentials -- which is the position
the evaluator is in.

    py -3.13 training/check_image_pullable.py
    py -3.13 training/check_image_pullable.py --file submission-ossp-skt.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCEPT = ",".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))


def _endpoint(repo: str):
    """(registry host, repository path, token URL) for an image reference."""
    parts = repo.split("/")
    if parts[0] == "ghcr.io":
        path = "/".join(parts[1:])
        return ("ghcr.io", path,
                f"https://ghcr.io/token?scope=repository:{path}:pull&service=ghcr.io")
    if parts[0] == "docker.io" or "." not in parts[0]:
        path = "/".join(parts[1:]) if parts[0] == "docker.io" else repo
        if "/" not in path:
            # A bare name resolves to docker.io/library/<name>, which is the
            # official-images namespace and will not hold a submission.
            path = "library/" + path
        return ("registry-1.docker.io", path,
                "https://auth.docker.io/token?service=registry.docker.io"
                f"&scope=repository:{path}:pull")
    return (parts[0], "/".join(parts[1:]), None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path,
                        default=REPO_ROOT / "submission-ossp-skt.json")
    args = parser.parse_args(argv)

    if not args.file.exists():
        print(f"{args.file} 이 없습니다.")
        return 1
    ref = json.loads(args.file.read_text(encoding="utf-8")).get("image_digest", "")
    if "@sha256:" not in ref:
        print(f"!! image_digest 가 registry/repository@sha256:... 형태가 아닙니다: {ref}")
        return 1
    repo, digest = ref.split("@", 1)
    host, path, token_url = _endpoint(repo)
    print(f"참조   {ref}")
    print(f"레지스트리 {host}   저장소 {path}")
    if "/" not in repo:
        print("!! 레지스트리 접두사가 없습니다. docker.io/library/ 로 해석되어 "
              "평가 코드가 받지 못할 가능성이 높습니다.")

    headers = {"Accept": ACCEPT}
    if token_url:
        try:
            token = json.load(urllib.request.urlopen(token_url, timeout=30)).get("token")
            headers["Authorization"] = f"Bearer {token}"
        except Exception as exc:
            print(f"!! 익명 토큰을 받지 못했습니다: {type(exc).__name__}")
            return 1

    request = urllib.request.Request(
        f"https://{host}/v2/{path}/manifests/{digest}", method="HEAD", headers=headers)
    try:
        urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as exc:
        print(f"!! 받을 수 없습니다 (HTTP {exc.code}).")
        if exc.code in (401, 403):
            print("   패키지가 비공개일 수 있습니다. ghcr 이면 공개로 바꾸십시오.")
        elif exc.code == 404:
            print("   그 다이제스트가 이 저장소에 없습니다. 태그가 아니라 "
                  "push 가 실제로 찍은 다이제스트인지 확인하십시오.")
        return 1
    except Exception as exc:
        print(f"!! 조회 실패: {type(exc).__name__}: {exc}")
        return 1

    print("OK: 자격증명 없이 받을 수 있습니다 -- 평가 코드와 같은 조건입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
