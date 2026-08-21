# ROCm Attention Stack — Architecture Decisions

## Base Image: rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0

| Layer | Version | Source |
|-------|---------|--------|
| ROCm  | 7.14.0  | Base image |
| PyTorch | 2.12.0+rocm7.14.0 | Base image |

**Why this image:**

1. **ROCm 7.14** was the latest stable ROCm release at build time. The user's 7900XTX (gfx1100, RDNA3) needs at least ROCm 6.x for full PyTorch support; 7.14 provides the most mature driver and runtime.

2. **PyTorch 2.12.0** is the latest PyTorch release paired with ROCm 7.14 in AMD's official Docker images. It includes:
   - `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` support for PyTorch SDPA Triton backend
   - `torch.backends.cuda.flash_sdp_enabled()` API for backend toggling
   - ROCm 7.14 kernel compatibility

3. **No other base image was viable**: The `rocm/pytorch` images are AMD's only officially maintained PyTorch+ROCm bundles. Custom PyTorch builds against a different ROCm version would require a full PyTorch rebuild.

---

## AOTriton 0.11.2

| Property | Value |
|----------|-------|
| Version | **0.11.2** (libaotriton_v2.so.0.11.2) |
| Installation | **Bundled with PyTorch — not independently installable** |

AOTriton is a **PyTorch-bundled** dependency — it ships inside the `torch` wheel and cannot be upgraded separately. You get whatever AOTriton version your PyTorch was compiled against.

PyTorch 2.12.0+rocm7.14.0 ships with AOTriton **0.11.2** (verified via `readlink -f libaotriton_v2.so`). This is NOT 0.12 as initially assumed from rocm-ninodes documentation comments.

Key limitation: AOTriton's default block sizes are optimized for CDNA (MI-series, 128+ KB shared memory). On RDNA3 (64 KB shared memory), AOTriton kernels are often suboptimal without explicit RDNA3 tuning.

---

## AITER v0.1.13

| Property | Value |
|----------|-------|
| Version | v0.1.13 (tag: `cdcfa833bd`) |
| Source | https://github.com/ROCm/aiter/tree/v0.1.13 |
| Build method | `pip install --no-build-isolation .` |
| Key arch | `GPU_ARCHS=gfx1100` (7900XTX) |
| Wheel format | Source build (no prebuilt cp312 wheel from AMD registry) |

**Why this version:**

### Version selection

AITER tags available at time of development: `v0.1.11`, `v0.1.12`, `v0.1.13`. Chose **v0.1.13** as the latest stable tag. The commit `cdcfa83` (v0.1.13) includes:

- `flash_attn_func` — RDNA3-optimized flash attention kernels
- `mla_forward` — Multi-head Latent Attention kernel (DeepSeek models, up to 17× decode speedup)
- Pre-compiled kernel support via `PREBUILD_KERNELS` env var

### Source build (not PyPI wheel)

AMD publishes wheels to two registries:
- **AMD PyPI**: `https://repo.radeon.com/rocm/manylinux/`
- **Google Cloud Storage**: `https://storage.googleapis.com/nm-public-pypi/simple/amd-aiter/`

Both only host **cp312** wheels matching our base image's Python 3.12. However, these prebuilt wheels link against AMD's internal ROCm paths and may not match the ROCm SDK Core layout in the PyTorch ROCm 7.14 image. Source build with `--no-build-isolation` was chosen for reliability.

### Build quirks discovered

| Attempt | Result | Cause |
|---------|--------|-------|
| `pip install .` (default) | ❌ Build isolation hides ROCm SDK | `rocm_sdk_core` not in isolated build env |
| `pip install --no-build-isolation .` | ✅ Works | Uses host packages including ROCm's venv wrappers |
| `pip install -e .` | ✅ Builds | But deleting source dir breaks module at runtime |
| `PREBUILD_KERNELS=2 pip install --no-build-isolation .` | ❌ Metadata gen fails | `setup.py` imports `jit` module before install completes |

### AITER JIT build fix: libamdhip64.so

AITER's `module_aiter_core` JIT build failed at runtime because the link step (`-L/opt/venv/lib -lamdhip64`) couldn't find `libamdhip64.so`. The PyTorch 2.12 + ROCm 7.14 image uses **ROCm SDK Core** as a pip package (`_rocm_sdk_core`), which ships `libamdhip64.so.7` inside `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/` — without the unversioned `.so` symlink that the linker expects.

**Fix in Dockerfile:**
```dockerfile
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib
RUN ln -sf /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7 /opt/venv/lib/libamdhip64.so
```

The `--no-build-isolation` install means AITER uses the pip-installed ROCm SDK Core rather than a system ROCm installation at `/opt/rocm/`.

### Why AITER over AOTriton on RDNA3

**Shared memory is the key constraint:**

| GPU | Architecture | Shared Memory/CU | AITER | AOTriton |
|-----|-------------|------------------|-------|----------|
| 7900XTX | RDNA3 (gfx1100) | **64 KB** | ✅ Tuned for 64 KB | ❌ Defaults to 128+ KB (CDNA) |
| MI250 | CDNA (gfx90a) | 128 KB | Works | ✅ Native target |

AITER's Triton kernels are explicitly tunable for RDNA3's 64 KB shared memory. AOTriton's pre-compiled kernels target CDNA's larger shared memory, leaving performance on the table for RDNA3.

MLA (Multi-head Latent Attention) for DeepSeek models is another AITER-exclusive advantage — the MLA decode kernel claims up to **17× speedup** over naive implementation on RDNA GPUs.

---

## ComfyKitchen

| Property | Value |
|----------|-------|
| Version | **0.2.30** (upgraded from 0.2.28) |
| Installation | `pip install --no-cache-dir comfy-kitchen==0.2.30` |
| Blacklist | Included in ComfyUI-Manager pip blacklist to prevent overwrite |

### Version 0.2.28 → 0.2.30 upgrade

**Original pin (0.2.28):** Version 0.2.28 was pinned because 0.2.30 had a verified bug on AMD ROCm gfx1100 GPUs — the `int8_convrot` operation produced solid-color (monochrome) images instead of correct output.

**Upgrade rationale (0.2.30):** User requested upgrade to 0.2.30 despite the known issue. The `int8_convrot` bug should be verified at runtime after rebuild.

**API compatibility patch:** ComfyUI v0.32.0 core `attention.py` calls `comfy_kitchen.int8_attention_is_available()` at import time. This API exists in 0.2.30 but not in 0.2.28. The Dockerfile applies a try/except guard to handle both versions:

```python
try:
    COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = comfy_kitchen.int8_attention_is_available()
except AttributeError:
    COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = False
```

---

## fa-rdna3 — 제거됨 (2026-08-01)

> **결정**: `chelokot/flash-attention-rdna3` v0.2.0 (fa-rdna3, 커스텀 노드 `RDNA3-Flash-Attention`) **이미지에서 제거**.

### 제거 사유 (실측 벤치 기준)

| 경로 | 활성화 | 시간 |
|------|--------|------|
| PyTorch SDPA (AOTriton) | 자동 | 22.04s |
| **fa-rdna3** | 워크플로 노드 | 20.67s |
| **FA2 Triton** | `--use-flash-attention` | **20.54s** |

- fa-rdna3(20.67s)와 FA2(20.54s)는 **성능 동급** (차이 0.13s = 노이즈)
- FA2가 ComfyUI **표준 경로** (`--use-flash-attention` 플래그, 우선순위 3번째)로 더 안전하고 유지보수 용이
- fa-rdna3의 유일한 장점이던 "워크플로 단위 선택 적용"은 이 워크플로(krea2 int8)에 필요 없음
- 커스텀 노드 유지 비용(빌드 시간, 잠재 충돌, `optimized_attention` 오버라이드 경쟁) 대비 이점 없음

### 이전 역할 (제거 전)

| Property | Value |
|----------|-------|
| Version | 0.2.0 (commit `a09f41d`) |
| Source | https://github.com/chelokot/flash-attention-rdna3 |
| Installation | Custom node clone + `pip install -e` |
| GPU target | gfx1100 only (7900 XT/XTX/GRE) |
| Dependencies | Triton only (no HIP C++ compilation) |

fa-rdna3는 gfx1100 전용 **순수 Triton** FlashAttention-2 구현이었음 (HIP C++ 확장·Composable Kernel 불필요). `ApplyRDNA3FlashAttention` 워크플로 노드 또는 `enable_rdna3_flash_attention()` SDPA 전역 패치로 활성화.

> ⚠️ 제거 전 경고: fa-rdna3 노드와 `--use-flash-attention`은 둘 다 `optimized_attention` 경로를 오버라이드 → **동시 사용 금지** (노드가 있으면 RDNA3 경로 우선). 이제 노드가 없으므로 충돌 가능성 소멸.

---

## SageAttention — FA2 대체 (2026-08-16)

> **결정**: FA2(`--use-flash-attention`) → **SageAttention v2.2.0** (`--use-sage-attention`) 전환. thu-ml/SageAttention **PR #381** (Scorp1o117 rocm-triton-support, pinned `6aa2622f`).

### 왜 PR #381인가 (V1 patientx 패치 기각)

| | V1 (guinmoon 휠 = patientx RDNA3 패치) | **V2 (PR #381)** |
|---|---|---|
| 버전 | sageattention-1 1.0.6 | **2.2.0 (main)** |
| HIP 지원 | 수동 패치 (waves_per_eu, BLOCK 32/16) | **setup.py HIP 자동 감지 → CUDA ext 스킵** |
| 검증 환경 | Windows ROCm 7.12 nightly | **ROCm 7.14 + torch 2.12.0 + triton 3.7.1 (본 이미지와 동일)** |
| RDNA 크래시 | 미보고 | **`num_stages=1`로 use-after-free(#365) 회피, RDNA3.5에서 2.8x 회복** |
| head_dim | 64/96/128 전용 | **64–128 패딩 지원** |

### 실측 벤치 (2026-08-16, RX 7900 XTX, 동일 CLI_ARGS + 워크플로우 전환마다 재기동)

| 워크플로우 | FA2 | SAGE | 차이 |
|---|---|---|---|
| Krea2 INT8 (이미지) | 8.31s | 8.02s | -3.5% (노이즈) |
| LTX 2.5 (5s 영상) | 87.0s | 81.3s | **-6.5%** |
| MiniMax H3 (5s 영상) | 303.6s | 192.2s | **-36.7%** |

- 시퀀스 길이에 비례해 이점 증가 (INT8 QK 양자화 효과)
- `--lowvram`은 sage 벤치에서 **제거** (no-lowvram + `--enable-dynamic-vram`이 Krea2 OOM 재현 없이 안정)

### 구현 노트

- Dockerfile: `pip install --no-build-isolation .` (소스 빌드, HIP 감지 시 순수 파이썬 설치는 아님 — setup.py가 ext 빌드만 스킵)
- pip_blacklist에 sageattention 추가 (Manager가 CUDA 버전 덮어쓰기 방지)
- ⚠️ **PR #381은 open 상태** — fork repo에 pinned commit 의존. 업스트림 merge 전까지 fork 유지 필요
- mask 있는 어텐션은 ComfyUI가 자동 pytorch fallback (안전)
- 상세 벤치: [`SAGEATTENTION_BENCHMARK.md`](SAGEATTENTION_BENCHMARK.md)

---

## Attention Backend Priority (실측 벤치 반영, 2026-08-01 → 2026-08-16 갱신)

### ComfyUI 내장 우선순위 (AMD ROCm)

ComfyUI는 CLI 플래그 + 설치 패키지로 attention 구현을 선택 (우선순위 높은 순, [attention.py](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/modules/attention.py#L776-L796)):

| Priority | Implementation | 활성화 조건 |
|----------|---------------|------------|
| 1 | Sage attention | sageattention 설치 + `--use-sage-attention` |
| 2 | xformers | ROCm xformers 설치 (자동, 플래그 불필요) |
| 3 | **Flash attention (FA2)** | flash-attn 설치 + **`--use-flash-attention`** |
| 4 | PyTorch SDPA | `--use-pytorch-cross-attention` 또는 자동감지 (ROCm + torch ≥2.7 + AOTriton) |
| 5 | Split attention | `--use-split-cross-attention` |
| 6 | Sub-quadratic | 기본 폴백 |

> ⚠️ **xformers는 설치만 되어도 자동 우선** (플래그보다 우선) — FA2/SDPA를 쓰려면 설치하지 말 것.
> 이 이미지에는 xformers 미설치 → FA2 경로 오버라이드 리스크 없음.

### 벤치 결과 (krea2 int8, RX 7900 XTX, 연속 생성 run2+)

| 경로 | 활성화 | 시간 | 비고 |
|------|--------|------|------|
| PyTorch SDPA | 자동 | 22.04s | 기준 (v0.28.2) |
| **FA2 Triton** | `--use-flash-attention` | 20.54s | v0.28.2에서 -6.8% 최적 |
| **SageAttention v2.2.0** | **`--use-sage-attention`** | **8.02s** | **v0.32.0 최종 — FA2 대체 (2026-08-16)** |

- SageAttention은 Krea2(이미지) -3.5%, LTX 2.5 -6.5%, MiniMax H3 **-36.7%** (FA2 대비)
- 이 이미지의 flash-attn은 **Triton-only 빌드** (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` 빌드 시 고정) → 런타임 env 토글 무차별
- FA2 첫 생성은 커널 초기화 워밍업 (~30s), 이후 연속 생성 안정

### 구성 요약

```
CLI_ARGS += --use-sage-attention --enable-dynamic-vram   (--cache-none 제거, --lowvram 제거)
```

---

## Memory Allocation (OOM Prevention)

The env var `PYTORCH_HIP_ALLOC_CONF` (ROCm equivalent of `PYTORCH_CUDA_ALLOC_CONF`) is set to:

```
expandable_segments:True,garbage_collection_threshold:0.8
```

**Why this matters:** Without `expandable_segments:True`, the HIP allocator uses a fixed-size block pool. When a 22B model (~11 GB INT8) is loaded alongside VAE, LoRA, and activations (~19.6 GB peak), the pool fragments — leaving 4+ GB "free" but no contiguous 16 MiB block for new allocations. `expandable_segments:True` grows the pool dynamically, eliminating fragmentation.

`garbage_collection_threshold:0.8` triggers aggressive GC when 80% of allocated memory is in use, preventing silent OOM during sampling.

> ⚠️ **Wrong env var note**: The common but incorrect `PYTORCH_ALLOC_CONF` is **not a valid env var** in any PyTorch version. ROCm requires `PYTORCH_HIP_ALLOC_CONF` and CUDA requires `PYTORCH_CUDA_ALLOC_CONF`.

---

## VRAM Management — Krea2 vs LTX (운영 확정 조합)

RX 7900 XTX 24GB 단일 GPU에서 두 워크플로우의 VRAM 전략이 **근본적으로 다르다**.

| | Krea2 (이미지) | LTX 2.3 (영상) |
|---|---|---|
| 모델 크기 | int8 13GB | int8 21GB (24GB의 87%) |
| 여유 VRAM | ~11GB | ~2-3GB |
| 해법 | `--lowvram` (CLIP만 CPU) | **청킹** (lowvram 무의미) |
| 검증 | 연속 4회 20.7s, OOM 없음 | V3 + 시간축 chunk, OOM 없음 |

**결정 사항:**

1. **`--lowvram` 채택 (Krea2)**: CLIP/텍스트 인코더만 CPU로 오프로드, UNet은 GPU 상주. reserve-vram은 UNet을 매 스텝 CPU 왕복시켜 3분+ 역효과 → 기각. GC threshold 0.5 실험도 OOM 미해결 → 0.8 유지.
2. **TensorParallelV3 (LTX)**: `comfyui_tensor_parallel_v3` 노드, `ffn_chunks=8`. 모델 내 112개 FFN(`ff`+`audio_ff`)을 래핑해 FFN 활성화 peak를 1/8로. `--lowvram`·offload와 안전 동작. KJNodes의 `LTXVChunkFeedForward`(chunks=2, 비디오 FFN만)보다 래핑 범위·chunks 수 모두 우위.
3. **MuseDirector 시간축 chunk (LTX)**: `chunk_duration_seconds=10`으로 시퀀스를 시간 축 분할 → 어텐션 시퀀스 반토막. V3(메모리 축)와 직교 관계로 함께 사용.
4. **`--disable-dynamic-vram` 유지 (AMD 필수)**: main.py 57행에서 `comfy_aimdo.control.init()`이 `is_nvidia()` 조건 없이 실행 — disable 없으면 AMD에서도 aimdo prefetch가 켜져 검은 이미지 재발.

**기각된 기법 (LTX 영상):**
- `PatchModelAddDownscale` (Deep Shrink): latent 해상도를 축소 → LTX는 RoPE freqs를 블록 루프 전 1회 계산하므로 다운스케일된 토큰과 위치 불일치 → 사용 불가.
- `ComfyUI-TiledDiffusion`: 4D 이미지 latent 전제, 공간(H,W)만 타일링. 영상은 5D latent + 시간 축이 시퀀스에 포함 → 적용 불가.
- `--highvram`: LTX 21GB + gemma 8.8GB = 30GB > 24GB → 로드 시점 OOM.

**운영 compose 환경변수** (docker-compose.yml 반영):
- `HIP_FORCE_DEV_KERNARG=1`, `HSA_FORCE_FINE_GRAIN_PCIE=1` — 과거 OOM 원인으로 의심돼 제거했으나 **실제 운영 컨테이너에서 켠 상태로 안정 동작 확인**되어 유지. (95df45a 커밋의 제거 결정은 롤백)
- `HSA_ENABLE_SDMA=0` — 듀얼 GPU 환경에서 SDMA 비활성
- `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` — 듀얼 GPU 안전장치
- `HIP_VISIBLE_DEVICES=0,1,2` — 7900 XTX ×2 + 내장 GPU 가시화

---

## Image Tag

```
rocm7.14-py3.12-torch2.12.0-triton3.7.1-fa2.8.3-aiter0.1.13-comfy0.32.0
```

All versioned components are listed in the tag for reproducibility:

| Component | Version |
|-----------|---------|
| ROCm | 7.14 |
| Python | 3.12 |
| PyTorch | 2.12.0 |
| Triton | 3.7.1 |
| flash-attn | 2.8.3.post1 |
| SageAttention | 2.2.0 (PR #381, 6aa2622f — 태그에 미포함, Dockerfile 참조) |
| AITER | v0.1.13 |
| ComfyUI | v0.32.0 (2026-08-09 v0.31.0 업그레이드 → v0.32.0) |
| ComfyKitchen | 0.2.30 (0.2.28 → 0.2.30 업그레이드) |

> fa-rdna3 0.2.0은 v0.2.0 태그(`...-rdna30.2.0-...`)에서 사용되다가 2026-08-01 벤치 결과로 제거됨 (FA2와 동급 성능).
> FA2는 2026-08-16 sage 도입 전 표준 경로였으며, 현재는 sage가 대체 (FA2 유지 — sage fallback 불필요 시 제거 후보).
