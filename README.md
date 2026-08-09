# ComfyUI ROCm - Triton & Flash Attention Optimized
### AMD GPU 최적화 ComfyUI Docker 이미지

**Latest Version**: `comfyui-rocm:rocm7.14-py3.12-torch2.12.0-triton3.7.1-fa2.8.3-aiter0.1.13-comfy0.31.0`

This project provides a specialized Docker container optimized for running ComfyUI on AMD hardware using **ROCm 7.14** and **PyTorch 2.12**. It features high-performance optimizations including Triton and Flash Attention specifically tuned for ROCm architectures, enabling advanced workflows like INT8 precision (Krea2) and video generation (LTX 2.3 Director).

---

## Features

*   **Optimized Performance**: Powered by ROCm 7.14 + PyTorch 2.12 with AOTriton 0.11.2.
*   **Advanced Kernels**: Full support for **Triton**, **Flash Attention** (FA2), and **AITER** (AMD Inference Toolkit).
*   **Specialized Workflow Support**:
    *   **Krea2 (Int8)**: High-speed inference with low memory footprint using specialized INT8 kernels.
    *   **LTX 2.3 Director**: Optimized for efficient video generation workloads on RDNA3 architecture.
*   **Pre-installed Custom Nodes**: Includes essential nodes for professional workflows:
    *   `ComfyUI-Manager`: Complete management of nodes and models.
    *   `ComfyUI-INT8-Fast-ROCM`: Accelerated integer quantization kernels.
    *   `rocm-ninodes`: Specialized AMD optimization utilities.
    *   `ComfyUI-GGUF`: Efficient model loading via GGUF format.
    *   `ComfyUI-KJNodes`: A vast collection of utility nodes.
    *   `ComfyUI-VideoHelperSuite`: Robust video import/export functionality.
    *   `rgthree-comfy`: Enhanced workspace organization and workflow control.

## Requirements

*   **Hardware**: AMD Radeon GPU (e.g., RX 7900 XTX / gfx1100 RDNA3 preferred)
*   **Driver**: Host machine must have the appropriate AMD ROCm kernel driver installed.
*   **Software**: Docker & Docker Compose environment.

## Build Notes

This Docker image uses the base `rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0` image which already includes PyTorch and ROCm-optimized Triton (3.7.1) with AOTriton 0.11.2 bundled.

flash-attn is installed with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, which builds **only the Triton backend** — the HIP C++ extension (`flash_attn_2_cuda`) targets CDNA-only architectures (gfx90a/gfx942) and is incompatible with RDNA3 (gfx1100). The Triton backend is the official AMD-supported path for consumer GPUs.

> ⚠️ **활성화 방법**: FA2 경로는 CLI 인자 `--use-flash-attention`이 있어야 활성화됩니다 (env만으로는 안 됨).
> 벤치 결과 (v0.28.2): SDPA 22.04s → FA2 20.54s (**-6.8%**), RDNA3 FA 20.67s와 동급. v0.31.0에서는 Krea2 INT8이 **8.08s**로 측정 (상세: `docs/BENCHMARK_RESULTS.md`).
> 런타임 `FLASH_ATTENTION_TRITON_AMD_ENABLE` env 토글은 빌드 시 고정되어 있어 **무차별**입니다.

**AITER** (AMD Inference Toolkit) v0.1.13 is built from source (`github.com/ROCm/aiter` tag `v0.1.13`) with `GPU_ARCHS=gfx1100` and `AITER_USE_SYSTEM_TRITON=1` to use the system Triton. AITER is the same version used in the `rocm-ninodes` reference image for 20%+ attention kernel performance.

⚠️ `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE` is intentionally **NOT** set globally — it causes import errors on RDNA3 with PyTorch 2.12.

**fa-rdna3는 제거됨** (v0.2.0, `chelokot/flash-attention-rdna3`): 실측 벤치에서 FA2(`--use-flash-attention`)와 성능 동급(20.67s vs 20.54s)으로 판명되어 중복 제거. FA2가 ComfyUI 표준 경로로 더 안전하고 전역 적용 가능. 상세: [`docs/ARCHITECTURE_DECISIONS.md`](docs/ARCHITECTURE_DECISIONS.md)

### ROCm Package Protection

This image includes multi-layer protection to prevent CUDA package overwrites during node installs or updates:

| Layer | File | Effect |
|-------|------|--------|
| 1 | `pip-constraints.txt` | Pins ROCm torch version at pip level |
| 2 | `pip_blacklist.list` | Blocks Manager from installing torch/triton/flash-attn/aiter |
| 3 | `config.ini` | Manager settings: `security_level=weak`, `model_download_by_agent=True` |

Even with Manager's full permissions enabled (`security_level=weak`), ROCm-optimized packages are protected from accidental CUDA overwrites.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/[your-username]/comfyui-rocm-triton-flashattn.git
cd comfyui-rocm-triton-flashattn

# 2. Prepare configuration
cp .env.example .env
# Edit .env as needed

# 3. Launch with Docker Compose
docker compose up -d
```

## Volume Mounts

| Target Path | Host Path | Purpose |
|---|---|---|
| `/workspace/models` | `/mnt/nvmedata/comfy/models` | Checkpoints, VAE, LoRA 등 모델 저장소 |
| `/workspace/custom_nodes` | `/mnt/nvmedata/comfy/custom_nodes` | 커스텀 노드 (실제 운영 시 활성) |
| `/workspace/output` | `/mnt/nvmedata/comfy/output` | 생성된 이미지/영상 저장 |
| `/workspace/input` | `/mnt/nvmedata/comfy/input` | 입력 이미지 |
| `/workspace/user/default/workflows` | `/mnt/nvmedata/comfy/user/default/workflows` | 워크플로우 프리셋 |
| `/root/.cache/huggingface/hub` | `/mnt/nvmedata/comfy/hf-hub` | HF 캐시 |
| `/root/.cache/torch/hub` | `/mnt/nvmedata/comfy/torch-hub` | torch hub 캐시 |

> 로컬 개발 시 `./models`, `./custom_nodes` 등 상대 경로로 교체 가능 (compose volumes 참조).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `COMFYUI_IMAGE` | `comfyui-rocm` | Docker image name |
| `COMFYUI_VERSION` | `latest` | Docker image version tag |
| `COMFYUI_PORT` | `8188` | The port exposed for Web UI access |
| `CLI_ARGS` | `--disable-pinned-memory --enable-manager --disable-dynamic-vram --disable-async-offload --use-flash-attention --lowvram` | ComfyUI 실행 인자 |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True,garbage_collection_threshold:0.8` | ROCm 메모리 단편화 방지 + 자동 GC (OOM 방지) |
| `PYTORCH_TUNABLEOP_ENABLED` | `0` | TunableOp 비활성 (ROCm에서 역효과) |
| `HIP_VISIBLE_DEVICES` | `0,1,2` | GPU 선택 (듀얼 컨테이너 시 instance별 분리) |
| `HSA_ENABLE_SDMA` | `0` | SDMA 가속 (듀얼 GPU면 0) |
| `HIP_FORCE_DEV_KERNARG` | `1` | HIP 커널 인자 device 메모리 강제 |
| `HSA_FORCE_FINE_GRAIN_PCIE` | `1` | PCIe fine-grain 메모리 |
| `NCCL_P2P_DISABLE` | `1` | 듀얼 GPU 안전장치 |
| `NCCL_IB_DISABLE` | `1` | InfiniBand 미사용 |
| `OMP_NUM_THREADS` | `8` | CPU 스레드 (호스트 코어 수에 맞게) |
| `MALLOC_MMAP_THRESHOLD_` | `65536` | Prevent glibc memory fragmentation OOM |
| `MALLOC_TRIM_THRESHOLD_` | `65536` | Prevent glibc memory fragmentation OOM |

## 최적 실행 설정 (벤치 결과 기반)

RX 7900 XTX (gfx1100) 기준 벤치: 연속 생성 27.55s → **8.08s (-71%)** (v0.31.0 실측, v0.28.2 대비 ~2.5배). 자세한 실험표는 [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md) 참고.
```
--disable-pinned-memory --enable-manager --disable-dynamic-vram --disable-async-offload --use-flash-attention --lowvram
```
- `--disable-dynamic-vram` (**필수**): comfy-aimdo DynamicVRAM prefetch의 in-place FP8 재양자화가 2회차부터 검은 이미지를 유발. 상세: [`docs/DEBUG_LOG_BLACK_IMAGES.md`](docs/DEBUG_LOG_BLACK_IMAGES.md)
- `--use-flash-attention`: FA2 Triton 경로 활성화 (SDPA 대비 -6.8%)
- `--lowvram`: **Krea2 반복 생성 OOM 해결** — CLIP/텍스트 인코더만 CPU로, UNet은 GPU 상주. 속도 20.7s 유지 (reserve-vram은 UNet thrash로 3분+ 역효과)
- `--cache-none` **제거**: entrypoint 오버라이드로 제거 필요 (매 생성마다 12.8GB 재로드 = +5.5s)

### 주의사항
- **`--disable-dynamic-vram` 없으면 검은 이미지 발생** — 절대 제거하지 말 것
- 컨테이너 `ulimits nofile: 1048576` 필수 (기본 1024면 `Too many open files`로 ComfyUI 사망)
- `--cache-none` 제거 시 모델이 RAM에 상주 (시스템 RAM 부족 시 `--cache-none` 복원 필요)
- `HIP_FORCE_DEV_KERNARG=1` / `HSA_FORCE_FINE_GRAIN_PCIE=1`: 과거 OOM 원인으로 의심돼 제거했으나, **실제 운영 컨테이너에서 켠 상태로 Krea2/LTX 모두 안정 동작** 확인. 보수적 기본값으로 compose에 유지.
- 디스크: `/var/crash`·`/var/lib/apport/coredump`의 Python 크래시 덤프가 수 GB 누적되어 디스크를 꽉 채울 수 있음. 주기적 정리 필요 (`sudo rm -f /var/crash/* /var/lib/apport/coredump/*`).

## Workflows

### Krea2 (Int8) Workflow
Leverage the pre-installed `ComfyUI-INT8-Fast-ROCM` to achieve massive speedups in image generation by using specialized quantization kernels. Simply load the INT8 optimized workflow template to begin.

### LTX 2.3 Director
High-performance video generation workflows are supported out-of-the-box through optimized Triton and Flash Attention implementation. Perfect for RDNA3 cards like the 7900 XTX.

**How to use:**
1.  Open the Web UI at `http://localhost:[YOUR_PORT]`.
2.  Drag and drop the downloaded `.json` workflow files into the browser window.

## Troubleshooting

*   **GPU not detected**: Check `docker logs <container_id>`. Ensure ROCm drivers are properly mapped.
*   **Out of Memory (OOM)**: Add `--lowvram` or `--novram` via the `CLI_ARGS` environment variable in your `.env`. The image sets `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8` to minimize fragmentation, but large models (22B+) may still need `--lowvram` on 24GB cards.
*   **LTX 영상 NaN 오디오 (ffmpeg `Input contains (near) NaN/+-Inf`)**: `UNETLoaderMultiGPU`의 `fp8_e4m3fn_fast` 옵션으로 int8 모델을 로드하면 fp8이 ROCm에서 emulated 연산이라 NaN이 발생. dtype을 `default`로 바꿀 것.
*   **LTX 22B VRAM 부족**: `TensorParallelV3Node` (ffn_chunks=8) + MuseDirector 시간축 chunk가 24GB 단일 GPU 기준. 상세: [`docs/VRAM_MANAGEMENT_NODES.md`](docs/VRAM_MANAGEMENT_NODES.md)
*   **Permission Issues**: Verify write permissions on your host volume directories (`models`, `output`).
*   **Log Inspection**: Use `docker logs -f <container_name>` for real-time debugging.

## License

This project is licensed under the **MIT License**.
