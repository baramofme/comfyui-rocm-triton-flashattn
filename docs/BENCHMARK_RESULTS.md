# ComfyUI ROCm — Performance Benchmark & Optimal Config

> 벤치마크일: 2026-08-01 · RX 7900 XTX (gfx1100) · torch 2.12.0+rocm7.14 · ComfyUI v0.28.2
> 워크플로: `image_krea2_turbo_t2i_int8` (krea2_turbo_int8_convrot, 1024×1024, 4-step euler, FP8 CLIP)
> 측정: 연속 생성 wall time (run2+ 안정값)

---

## 벤치마크 결과

| # | 설정 | 연속 생성 (run2+) | 판정 |
|---|------|------------------|------|
| baseline | `--cache-none` + 어텐션 env OFF 3종 | 27.55s | 기준 |
| 1 | 모든 성능 기능 활성 (HIPBLASLT+TunableOp+MIOpen+FLASH) | 27.62s | 무차별 |
| 2 | baseline + `TORCH_BLAS_PREFER_HIPBLASLT=1` | 27.58s | 무차별 |
| 3 | baseline 재현 (오차 확인) | 27.63s | - |
| A | baseline + SDPA EFFICIENT 강제 | 27.57s | 무차별 |
| **B** | **`--cache-none` 제거 (모델 캐시 활성)** | **22.05s** | **🏆 -20%** |
| C | B + HIPBLASLT=1 | 22.04s | 무차별 |
| D | B + TunableOp=1 | 23.55s | **역효과 (+1.5s)** |
| E | B + 어텐션 env 기본값 | 22.05s | 무차별 |
| **F** | **B + 최소 env (캐시만)** | **22.04s** | **🏆 (SDPA) 최종** |
| G | F + RDNA3 FA 노드 (fa-rdna3) ~~이미지에서 제거됨~~ | 20.67s | 🏆 -6% |
| **H** | **F + `--use-flash-attention` (FA2, env OFF)** | **20.54s** | **🏆 최종 -6.8%** |
| I | H + `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (FA2 Triton) | 20.63s | 동급 |

## 결론

### 유일한 실질 개선: `--cache-none` 제거
- `--cache-none`은 매 생성마다 12.8GB 모델을 디스크에서 재로드 → **5.5s 낭비**
- 제거 시 연속 생성 **27.55 → 22.04s (-20%)**, 5회 반복 표준편차 0.005s (매우 안정)
- 단점: RAM 사용 증가 (모델 상주). 시스템 RAM 100GB라 문제 없음

### FA 경로 2종 (SDPA 대비 ~6-7% 추가 개선)
- **FA2 env OFF** (20.54s) / **FA2 env TRUE** (20.63s) — 동급 (~0.1s 차이는 노이즈)
- **RDNA3 FA 노드 (fa-rdna3)** 는 벤치상 FA2와 동급(20.67s)이었으나 **중복으로 이미지에서 제거됨** (2026-08-01). FA2가 표준 경로.
- **FA2 경로 활성화**: `--use-flash-attention` CLI 인자 필요 (DAO-AILab flash-attn 패키지 필수)
  - 이 이미지의 flash-attn은 빌드 시 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`로 **Triton-only 빌드** — 실험 H/I 모두 Triton 경로
  - `FLASH_ATTENTION_TRITON_AMD_ENABLE` 런타임 env 토글은 **무차별** (빌드 시 고정되어 있기 때문)
  - FA2 첫 생성은 워밍업으로 ~30s (커널 초기화) — 이후 연속 생성 안정
- **이전 "어텐션 env 3종 no-op" 판정은 플래그 미설정으로 FA2 경로가 비활성이었기 때문** — env 자체가 아니라 `--use-flash-attention`이 핵심 스위치

### 무차별 (제거해도 됨)
- `TORCH_BLAS_PREFER_HIPBLASLT=1` (부팅 로그 권장이지만 이 워크플로에선 무영향)
- `AITER_TRITON_ONLY` / `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` (FA2 경로에선 무영향)
- SDPA EFFICIENT 강제 (prestartup) — 마이크로벤치 50배 차이가 실워크플로엔 무영향
- `MIOPEN_FIND_ENFORCE=0` / `MIOPEN_DEBUG_DISABLE_FIND_DB=1`

### 역효과
- **`PYTORCH_TUNABLEOP_ENABLED=1`: +1.5s** (첫 실행 커널 튜닝 오버헤드) — **0 유지**

### 필수 유지 (버그 방지)
- `--disable-dynamic-vram`: **검은 이미지 버그 해결** (2회차 NaN, comfy-aimdo prefetch)
- `--disable-async-offload`: 무해 (성능 무차별)

### 부수 발견: fd limit
- 컨테이너 기본 fd limit 1024가 너무 낮아 ComfyUI가 `Too many open files: '/proc/meminfo'`로 죽을 수 있음
- **`ulimits: nofile: 1048576` 필수**

### Krea2 반복 생성 OOM → `--lowvram` 해결 (2026-08-01)
- 반복 생성 시 OOM 발생 (`--disable-dynamic-vram`으로 모델 19.2GB 상주 + INT8 chunk 임시 텐서 128MB 할당 실패)
- `--reserve-vram 1.5`: UNet 매 스텝 CPU 왕복 → KSampler 3분+ **역효과, 기각**
- GC threshold 0.5: OOM 미해결 → 0.8 복귀
- **`--lowvram` 채택**: CLIP/텍스트 인코더만 CPU, UNet GPU 상주 → 연속 4회 **20.70/20.66/20.73/20.67s** OOM 없음 (속도 저하 없음)

### LTX 2.3 영상 검증 (2026-08-01)
- **TensorParallelV3** (`ffn_chunks=8`, 112개 FFN 래핑): 24GB 단일 GPU에서 OOM 없이 성공
- **MuseDirector 시간축 chunk** (`chunk_duration_seconds=10`): single-pass(1 chunk) OOM → chunked(2 chunks) 성공
- 1280×704, 288프레임 기준. 8-step 기준 ~10-18s/it (시퀀스 길이 의존)

---

## 최적 설정 (Dokploy UI 반영용)

### 1. `--cache-none` 제거 — entrypoint 오버라이드

이미지의 `/opt/entrypoint.sh`가 `--cache-none`을 하드코딩하므로, 오버라이드 파일을 마운트:

**호스트에 파일 생성** (`/mnt/nvmedata/comfy/entrypoint_nocache.sh` — 이미 생성됨):
```bash
#!/bin/bash
set -e
mkdir -p /workspace/user/__manager
printf "[default]\nmodel_download_by_agent = True\nsecurity_level = weak\nnetwork_mode = personal_cloud\nallow_git_url_install = True\nallow_pip_install = True\n" > /workspace/user/__manager/config.ini
printf "torch\ntriton\nflash-attn\naiter\n" > /workspace/user/__manager/pip_blacklist.list
exec python main.py \
    --listen 0.0.0.0 \
    --port ${PORT:-8188} \
    --disable-api-nodes \
    --disable-mmap \
    --enable-manager \
    --enable-manager-legacy-ui \
    ${CLI_ARGS}
```

### 2. docker-compose.yml (Dokploy 반영)

```yaml
services:
  comfyui-rocm:
    image: comfyui-rocm:rocm7.14-py3.12-torch2.12.0-triton3.7.1-fa2.8.3-aiter0.1.13-comfy0.28.2
    container_name: comfyui_gpu0
    restart: unless-stopped
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp:unconfined
    ports:
      - 8189:8188
    entrypoint: ["/opt/entrypoint.sh"]
    ulimits:
      nofile:
        soft: 1048576
        hard: 1048576
    environment:
      - CLI_ARGS=--disable-pinned-memory --enable-manager --disable-dynamic-vram --disable-async-offload --use-flash-attention --lowvram
      - OMP_NUM_THREADS=8
      - MKL_NUM_THREADS=8
      - OPENBLAS_NUM_THREADS=8
      - VECLIB_MAXIMUM_THREADS=8
      - NUMEXPR_NUM_THREADS=8
      - HSA_ENABLE_SDMA=0
      - HIP_VISIBLE_DEVICES=0,1,2
      - NCCL_P2P_DISABLE=1
      - NCCL_IB_DISABLE=1
      - HIP_FORCE_DEV_KERNARG=1
      - HSA_FORCE_FINE_GRAIN_PCIE=1
      - PYTORCH_TUNABLEOP_ENABLED=0
      - PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
    volumes:
      - /mnt/nvmedata/comfy/entrypoint_nocache.sh:/opt/entrypoint.sh
      - /mnt/nvmedata/comfy/hf-hub:/root/.cache/huggingface/hub
      - /mnt/nvmedata/comfy/torch-hub:/root/.cache/torch/hub
      - /mnt/nvmedata/comfy/models:/workspace/models
      - /mnt/nvmedata/comfy/input:/workspace/input
      - /mnt/nvmedata/comfy/output:/workspace/output
      - c8a836a0937642172cadb5f321415f1a73303dc800d300ed58ad58c1edb32424:/workspace/user
    networks:
      - dokploy-network
volumes:
  c8a836a0937642172cadb5f321415f1a73303dc800d300ed58ad58c1edb32424:
    external: true
networks:
  dokploy-network:
    external: true
```

### 변경 요약 (기존 대비)

| 항목 | 기존 | 최적 |
|------|------|------|
| `--cache-none` | 강제 (entrypoint) | **제거** (entrypoint 오버라이드) |
| `--use-flash-attention` | 없음 (SDPA) | **추가 (FA2 Triton, -6.8%)** |
| `--lowvram` | 없음 | **추가** (Krea2 반복 OOM 해결, 속도 무손실) |
| `PYTORCH_TUNABLEOP_ENABLED` | 0 | 0 (1이면 역효과) |
| 어텐션 env 3종 | FALSE/0/0 | 제거해도 무방 (`--use-flash-attention`가 핵심) |
| HIPBLASLT | 없음 | 제거해도 무방 |
| ulimits nofile | 1024 (기본) | **1048576 (필수)** |
| `--disable-dynamic-vram` | 있음 | **유지 (버그 방지)** |

### 주의
- `/workspace/user`는 named volume `c8a836a0...` — bind mount로 바꾸면 워크플로/설정이 사라짐. 반드시 외부 볼륨 유지
- prestartup SDPA EFFICIENT 강제는 성능 무차별이므로 제거해도 됨
- ~~`ApplyRDNA3FlashAttention` 노드 충돌~~ → fa-rdna3가 이미지에서 제거됨 (2026-08-01) — 충돌 가능성 소멸
