# ComfyUI ROCm - Triton & Flash Attention Optimized
### AMD GPU 최적화 ComfyUI Docker 이미지

**Latest Version**: `comfyui-rocm:rocm7.2.4-py2.10-fa2.8`

This project provides a specialized Docker container optimized for running ComfyUI on AMD hardware using **ROCm 7.2.4** and **PyTorch 2.10**. It features high-performance optimizations including Triton and Flash Attention specifically tuned for ROCm architectures, enabling advanced workflows like INT8 precision (Krea2) and video generation (LTX 2.3 Director).

---

## Features

*   **Optimized Performance**: Powered by ROCm 7.2.4 + PyTorch 2.10.
*   **Advanced Kernels**: Full support for **Triton** and **Flash Attention** optimized for AMD GPUs.
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

This Docker image uses the base `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` image which already includes PyTorch and ROCm-optimized Triton. An alternative approach ([gist](https://gist.github.com/alexheretic/d868b340d1cef8664e1b4226fd17e0d0)) installs torch from `rocm.nightlies.amd.com` and reinstalls it after flash-attn to restore ROCm Triton. We chose the base image approach because:

*   **Faster builds**: No need to download/reinstall ~2-3GB torch package twice
*   **Simpler**: Save/restore ROCm triton binary is faster than full torch reinstall
*   **Same result**: Both approaches achieve ROCm-optimized Triton with flash-attn support

### ROCm Package Protection

This image includes multi-layer protection to prevent CUDA package overwrites during node installs or updates:

| Layer | File | Effect |
|-------|------|--------|
| 1 | `pip-constraints.txt` | Pins ROCm torch/triton versions at pip level |
| 2 | `pip_blacklist.list` | Blocks Manager from installing torch/triton/flash-attn |
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
| `AITER_TRITON_ONLY` | `1` | Prevents aiter JIT compilation (required for stability) |
| `COMFYUI_ENABLE_MIOpen` | `1` | Enable MIOpen for better upscaling performance |
| `MIOPEN_FIND_MODE` | `FAST` | MIOpen kernel find mode (FAST for initial speed) |
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
