<!--
SPDX-FileCopyrightText: Copyright 2026 difficcd
SPDX-License-Identifier: Apache-2.0
-->

# 네이티브 리눅스 실행 시간 측정 — 인계 문서

이 문서만 읽고 작업할 수 있게 썼습니다. 사전 맥락은 필요 없습니다.

---

## 무엇을 왜 재는가

SKT "Efficient LLM Routing Challenge" 제출물입니다. 프롬프트만 보고 세 개의
LLM 중 하나를 고르는 라우터를 컨테이너로 제출하며, 공식 채점 환경은
**linux/arm64, CPU 2개, 메모리 2GiB, 네트워크 차단, 등급당 90초**입니다.

Windows(Docker Desktop)에서 측정했더니 세 등급 모두 90초를 넘겼습니다.
그런데 그 환경은 x86 위에서 **QEMU로 arm64를 에뮬레이션**하고, 그 위에
Docker Desktop VM과 9p 파일시스템이 더 얹힙니다. 같은 이미지를 네이티브
amd64로 돌리면 **2,640문항에 14.9초**였습니다.

**그래서 알고 싶은 것은 하나입니다: 에뮬레이션 층이 없는 실제 리눅스에서
몇 초인가?**

- 이 장비가 **aarch64**면 → 공식 환경과 같은 아키텍처. **실측값**이 됩니다
- 이 장비가 **x86_64**면 → 대리값. 그래도 QEMU·VM 층이 빠지므로 유용합니다

---

## 준비물

1. 이 저장소 (`git clone https://github.com/difficcd/SmartRouter`)
2. Docker (또는 podman — `--docker-command` 없이도 `docker` 이름이면 됩니다)
3. **입력 파일 2개** — 저장소에 없습니다(`.gitignore` 대상). 두 방법 중 하나:

   **(a) 파일을 전달받은 경우** — USB/scp 로 받은 `native-kit/` 을 배치:
   ```bash
   mkdir -p data/materialized/dev data/materialized/train
   cp <받은경로>/dev-inputs.json   data/materialized/dev/inputs.json
   cp <받은경로>/train-inputs.json data/materialized/train/inputs.json
   ```
   무결성 확인 (SHA-256 앞 16자리):
   ```
   dev-inputs.json     3,915,215바이트   5920f9ea9e3da147
   train-inputs.json   7,865,187바이트   029a0fb1f70432a0
   ```

   **(b) 직접 만드는 경우** — 네트워크가 필요하고 시간이 걸립니다:
   ```bash
   python3 -m venv .venv-data
   .venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt
   .venv-data/bin/python tools/materialize_public_data.py
   ```

---

## 실행

```bash
bash training/native_timing.sh
```

스크립트가 `uname -m` 으로 아키텍처를 판별해 플랫폼을 정하고, 이미지를 빌드한
뒤 세 등급(fast/balanced/premium)을 **공식 자원 한도 그대로** 실행합니다.

Dev 880문항이 기본이고, Train+Dev 2,640문항까지 재려면:

```bash
bash training/native_timing.sh --combined
```

---

## 무엇을 보고할 것인가

스크립트 출력 전체를 그대로 전달해 주세요. 특히 이 네 가지:

1. **`uname -m` 결과** (aarch64 / x86_64) — 실측이냐 대리값이냐를 가릅니다
2. **등급별 초** 와 90초 대비 판정
3. **문항 수** (880 인지 2,640 인지)
4. **CPU 모델명** — `lscpu | grep 'Model name'`. 공식 장비와 성능 급이 다르면
   해석이 달라집니다

---

## 예상 결과와 이상 신호

| 상황 | 해석 |
|---|---|
| 880문항 5~15초 | **정상.** 예상대로이고 한도에 큰 여유 |
| 2,640문항 15~40초 | **정상** |
| 880문항 60초 이상 | **이상.** 에뮬레이션이 아직 걸려 있거나(플랫폼 확인) 자원 제한이 잘못 적용된 것 |
| 종료코드 0 아님 | 아래 문제 해결 참고 |

---

## 문제 해결

**`exec format error`** — 이미지 아키텍처와 CPU가 다릅니다. 스크립트는
`uname -m`에 맞춰 빌드하므로 정상이라면 나지 않습니다. 났다면
`docker buildx ls` 로 해당 플랫폼이 있는지 확인하세요.

**`No such image: sha256:...`** — buildx 가 attestation 이 붙은 매니페스트
리스트를 만들면 content ID 로 실행이 안 됩니다. 스크립트는
`--provenance=false --sbom=false` 를 이미 씁니다.

**`파일을 읽을 수 없습니다`** — 볼륨 마운트 경로 문제입니다. 스크립트는
절대 경로를 씁니다. SELinux 환경(Rocky/Fedora)이면 `-v` 뒤에 `:z` 가
필요할 수 있습니다:
```bash
# native_timing.sh 안의 -v 인자 두 곳에 :z 를 덧붙이세요
-v "$(pwd)/$IN:/challenge/input/inputs.json:ro,z"
```

**입력 파일이 없다고 나옴** — 위 "준비물 3" 을 하지 않았습니다.

---

## 하지 말아야 할 것

- 이미지나 소스를 **수정하지 마세요.** 제출본과 다른 것을 재면 의미가 없습니다
- 자원 제한 인자(`--cpus 2 --memory 2g --network none --read-only`)를
  **바꾸지 마세요.** 공식 조건입니다
- 결과가 나쁘게 나와도 **고치려 하지 마세요.** 그대로 보고하면 됩니다.
  나쁜 숫자도 정보입니다
