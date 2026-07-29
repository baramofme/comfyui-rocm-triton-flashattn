# ComfyUI ROCm - Triton & Flash Attention Optimized
### AMD GPU 최적화 ComfyUI Docker 이미지

**Latest Version**: `comfyui-rocm:rocm7.14-py3.12-torch2.12.0-triton3.7.1-fa2.8.3-aiter0.1.13-rdna30.2.0-comfy0.28.2`

This project provides a specialized Docker container optimized for running ComfyUI on AMD hardware using **ROCm 7.14** and **PyTorch 2.12**. It features high-performance optimizations including Triton and Flash Attention specifically tuned for ROCm architectures, enabling advanced workflows like INT8 precision (Krea2) and video generation (LTX 2.3 Director).

---

## Features

*   **Optimized Performance**: Powered by ROCm 7.14 + PyTorch 2.12 with AOTriton 0.13+.
*   **Advanced Kernels**: Full support for **Triton**, **Flash Attention** and **AITER** (AMD Inference Toolkit) optimized for AMD GPUs.
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

This Docker image uses the base `rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0` image which already includes PyTorch and ROCm-optimized Triton.

flash-attn is installed with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, which builds **only the Triton backend** — the HIP C++ extension (`flash_attn_2_cuda`) targets CDNA-only architectures (gfx90a/gfx942) and is incompatible with RDNA3 (gfx1100). The Triton backend is the official AMD-supported path for consumer GPUs.

**AITER** (AMD Inference Toolkit) v0.1.13 is built from source (`github.com/ROCm/aiter` tag `v0.1.13`) with `GPU_ARCHS=gfx1100` and `AITER_USE_SYSTEM_TRITON=1` to use the system Triton. AITER is the same version used in the `rocm-ninodes` reference image for 20%+ attention kernel performance.

`FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE` is set globally to automatically tune flash-attn Triton kernels for your specific GPU, providing an additional 5-10% performance gain on first use.

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
| `/workspace/models` | `./models` | Store Checkpoints, VAE, LoRA, etc. |
| `/workspace/custom_nodes` | `./custom_nodes` | Persist additional custom nodes you install |
| `/workspace/output` | `./output` | Saves all generated images and videos |
| `/workspace/user` | `./workflows` | Stores workflow presets and user workspace |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `COMFYUI_IMAGE` | `comfyui-rocm` | Docker image name |
| `COMFYUI_VERSION` | `latest` | Docker image version tag |
| `COMFYUI_PORT` | `8188` | The port exposed for Web UI access |
| `CLI_ARGS` | `--use-flash-attention` | Extra command line arguments passed to ComfyUI |
| `COMFYUI_MODELS` | `./models` | Host path for model storage |
| `COMFYUI_CUSTOM_NODES` | `./custom_nodes` | Host path for additional custom nodes |
| `COMFYUI_OUTPUT` | `./output` | Host path for generated output |
| `COMFYUI_WORKFLOWS` | `./workflows` | Host path for workflow presets |
| `COMFYUI_ENABLE_MIOpen` | `1` | Enable MIOpen for better upscaling performance |
| `MIOPEN_FIND_MODE` | `FAST` | MIOpen kernel find mode (FAST for initial speed) |
| `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE` | `TRUE` | Enable flash-attn Triton autotuning for 5-10% perf uplift |
| `AITER_TRITON_ONLY` | `1` | Use only Triton kernels for AITER (avoids CDNA-only HIP kernels) |
| `MALLOC_MMAP_THRESHOLD_` | `65536` | Prevent glibc memory fragmentation OOM |
| `MALLOC_TRIM_THRESHOLD_` | `65536` | Prevent glibc memory fragmentation OOM |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | Optimizes PyTorch memory allocation for ROCm |

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
*   **Out of Memory (OOM)**: Add `--lowvram` or `--novram` via the `COMFYUI_ARGS` environment variable in your `.env`.
*   **Permission Issues**: Verify write permissions on your host volume directories (`models`, `output`).
*   **Log Inspection**: Use `docker logs -f <container_name>` for real-time debugging.

## License

This project is licensed under the **MIT License**.
