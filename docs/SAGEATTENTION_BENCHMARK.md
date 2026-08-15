# SageAttention (PR #381, v2.2.0) ROCm 벤치 결과 — RX 7900 XTX

> 벤치일: 2026-08-16 · RX 7900 XTX (gfx1100) · torch 2.12.0+rocm7.14 · triton 3.7.1 · ComfyUI v0.32.0
> SageAttention: thu-ml/SageAttention **PR #381** (ROCm/HIP 지원) — v2.2.0, HIP 자동 감지 + Triton 경로 + `num_stages=1`
> 설치: 공식 레포 main(d1a57a54 == PR base) clone → `pull/381.diff | git apply` → `pip install .`
> 비교: `--use-sage-attention` vs `--use-flash-attention` (FA2 Triton-only 빌드). 동일 CLI_ARGS (--disable-pinned-memory --enable-manager --force-non-blocking --enable-dynamic-vram), 각 워크플로우 전환마다 컨테이너 재기동 (모델 VRAM 상주 영향 배제)

---

## 결과 요약

| 워크플로우 | FA2 | SAGE | 차이 |
|---|---|---|---|
| **Krea2 INT8** (1024×1024, 4-step euler) | 8.31s | 8.02s | **-3.5%** (노이즈 수준) |
| **LTX 2.5** (448×832, 5s, 24fps, Stage1 8-step + Stage2 3-step, INT8 ConvRot + Q3 GGUF) | 87.0s | 81.3s | **-6.5%** |
| **MiniMax H3** (9:16 0.4MP, 5s, 8-step, FL2VA Q4_K_M GGUF + TensorParallelV3) | 303.6s | 192.2s | **-36.7%** |

- Krea2/LTX/MiniMax 모두 sage 우위. 영상 생성일수록 시퀀스 길이가 길어 **sage의 INT8 QK 양자화 이점이 커짐**
- MiniMax H3의 -36.7%는 sage가 FA2보다 빠를 뿐 아니라, 이 모델에서 어텐션이 전체 시간의 큰 비중을 차지하기 때문

---

## 상세 측정

### Krea2 INT8 (이미지, 6회)
| 백엔드 | run1 | run2 | run3 | run4 | run5 | run6 | median |
|---|---|---|---|---|---|---|---|
| FA2 | 0.5 (캐시) | 8.52 | 8.52 | 8.02 | 8.10 | 8.52 | 8.31 |
| SAGE | 0.0 (캐시) | 8.02 | 8.52 | 8.02 | 8.21 | 8.02 | 8.02 |

### LTX 2.5 (warmup + run)
| 백엔드 | warmup (모델 로드 포함) | run | 비고 |
|---|---|---|---|
| FA2 | 114.1s | **87.0s** | |
| SAGE | 224.5s | **81.3s** | warmup은 캐시 히트로 3s, run 81.3s |

### MiniMax H3 (warmup + run)
| 백엔드 | warmup (모델 로드 포함) | run | 비고 |
|---|---|---|---|
| FA2 | 418.2s | **303.6s** | |
| SAGE | 318.2s | **192.2s** | |

---

## 구현 요약 (이미지 반영)

- **Dockerfile**: SageAttention v2.2.0 — 공식 `thu-ml/SageAttention` clone (main `d1a57a54` == PR #381 base) → `curl pull/381.diff | git apply` → `pip install --no-build-isolation .`
  - setup.py가 `torch.version.hip` 감지 → CUDA ext 빌드 스킵, 순수 Triton
  - HIP에서 `num_stages=1` 강제 → AMD Triton pipelining use-after-free(#365) 회피, RDNA3.5 기준 2.8x 회복
- **pip_blacklist**: sageattention 추가 (Manager가 CUDA 버전으로 덮어쓰지 못하게)
- **Dokploy compose**: CLI_ARGS에 `--use-sage-attention` 사용 중 (사용자 반영)

### 주의사항
- PR #381은 **아직 open 상태** (merge 안 됨). 공식 레포 base commit + `pull/381.diff`로 재현 가능 — fork 의존 없음
- PR이 업스트림에 merge되면 Dockerfile의 base commit만 main HEAD로 교체하면 됨
- ComfyUI v0.32.0은 `--use-sage-attention` 공식 지원. sageattention 미설치 시 exit(-1) → 패키지 필요
- mask 있는 어텐션은 ComfyUI가 자동 pytorch fallback (안전)
- 이전 V1 접근(guinmoon 휠 = patientx RDNA3 패치)은 **사용하지 않음** — V2(PR #381)가 같은 환경(ROCm 7.14/torch 2.12/triton 3.7.1)에서 실측됐고, head_dim 64-128 패딩, bf16 지원, HIP 자동 감지로 더 견고
