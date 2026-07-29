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

## fa-rdna3 v0.2.0 (chelokot/flash-attention-rdna3)

| Property | Value |
|----------|-------|
| Version | 0.2.0 (commit `a09f41d`) |
| Source | https://github.com/chelokot/flash-attention-rdna3 |
| Installation | Custom node clone + `pip install -e` |
| GPU target | gfx1100 only (7900 XT/XTX/GRE) |
| Dependencies | Triton only (no HIP C++ compilation) |

**Why add fa-rdna3 alongside AITER:**

fa-rdna3 provides a **pure Triton** FlashAttention-2 implementation purpose-built for RDNA3, with no dependency on HIP C++ extensions or Composable Kernel. This avoids the JIT compilation issues that AITER encounters on newer ROCm releases.

Key advantages over AITER:

| Feature | AITER | fa-rdna3 |
|---------|-------|----------|
| HIP C++ build | Required (JIT, ~21s) | **None** — Triton only |
| Forward vs AOTriton | Unknown | **Up to 1.79×** |
| Backward vs AOTriton | Unknown | **Up to 1.63×** |
| Split-K decode | Not available | **Up to 13.4×** over serial |
| torch.compile | ❌ (opaque op) | ✅ (torch.library.custom_op) |
| ComfyUI node | Not available | `ApplyRDNA3FlashAttention` |
| SDPA monkey-patch | Not available | `enable_rdna3_flash_attention()` |
| gfx1100 validation | General ROCm | **Explicit gfx1100 verification** |

fa-rdna3 is installed as a **ComfyUI custom node** (`RDNA3-Flash-Attention`) and as a pip editable package. Users opt in by adding the `ApplyRDNA3FlashAttention` node to their workflow (model_patches), or enable globally via `enable_rdna3_flash_attention()`.

---

## Attention Backend Priority

```
┌──────────────────────────────────────────────────────┐
│  1. fa-rdna3 RDNA3 Flash Attention (model_patches)  │ ← Pure Triton, gfx1100 tuned
│     Condition: user adds ApplyRDNA3FlashAttention    │    up to 1.79× vs AOTriton
│     node to workflow                                  │
├──────────────────────────────────────────────────────┤
│  2. AITER flash_attn (monkey-patch SDPA)            │ ← RDNA3 optimized, 64 KB shared mem
│     Condition: arch ∈ {rdna3, rdna3_5}               │
├──────────────────────────────────────────────────────┤
│  3. flash-attn Triton backend (env var)              │ ← ROCm 6.4+ / PyTorch SDPA built-in
│     Condition: flash_attn package installed          │
├──────────────────────────────────────────────────────┤
│  4. AOTriton (PyTorch built-in)                     │ ← Always available, CDNA-optimized
│     Version: 0.11.2                                   │
└──────────────────────────────────────────────────────┘
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

## Image Tag

```
rocm7.14-py3.12-torch2.12.0-triton3.7.1-fa2.8.3-aiter0.1.13-rdna30.2.0-comfy0.28.2
```

All versioned components are listed in the tag for reproducibility:

| Component | Version |
|-----------|---------|
| ROCm | 7.14 |
| Python | 3.12 |
| PyTorch | 2.12.0 |
| Triton | 3.7.1 |
| flash-attn | 2.8.3.post1 |
| AITER | v0.1.13 |
| fa-rdna3 | 0.2.0 |
| ComfyUI | v0.28.2 |
