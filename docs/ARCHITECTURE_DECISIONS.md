# ROCm Attention Stack — Architecture Decisions

## Base Image: rocm/pytorch:rocm7.14_ubuntu24.04_py3.13_pytorch_release_2.12.0

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

## AOTriton 0.12.0

| Property | Value |
|----------|-------|
| Version | 0.12.0 (with PyTorch 2.12.0) |
| Installation | **Bundled with PyTorch — not independently installable** |
| Triton base | Triton 0.12 |

**Why this version:**

AOTriton is a **PyTorch-bundled** dependency — it ships inside the `torch` wheel and cannot be upgraded separately. You get whatever AOTriton version your PyTorch was compiled against.

PyTorch 2.12.0 ships with AOTriton 0.12.0 (Triton 0.12-based). Key improvement over earlier versions:

- **Block-IO coalescing** (Triton 0.12 feature): Up to **+26.5% attention speedup** on RDNA GPUs by reducing global memory transactions
- AOTriton < 0.12 was based on Triton < 0.12, which lacked this optimization

The `attention.py` `_check_aotriton_version()` function detects AOTriton < 0.12 and logs a warning to upgrade PyTorch.

> NOTE: AOTriton's default block sizes are optimized for CDNA (MI-series, 128+ KB shared memory). On RDNA3 (64 KB shared memory), the Triton compiler auto-tunes to smaller blocks. AITER's Triton kernels handle this explicitly with RDNA3-specific tuning, which is why AITER takes priority over AOTriton in our stack.

---

## AITER v0.1.13

| Property | Value |
|----------|-------|
| Version | v0.1.13 (tag: `cdcfa833bd`) |
| Source | https://github.com/ROCm/aiter/tree/v0.1.13 |
| Build method | `pip install --no-build-isolation .` |
| Key arch | `GPU_ARCHS=gfx1100` (7900XTX) |
| Wheel format | Source build (no prebuilt cp313 wheel) |

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

Both only host **cp312** wheels. Our base image runs **cp313** (Python 3.13). No prebuilt wheel exists for cp313, so source build was mandatory.

### Build quirks discovered

| Attempt | Result | Cause |
|---------|--------|-------|
| `pip install .` (default) | ❌ Build isolation hides ROCm SDK | `rocm_sdk_core` not in isolated build env |
| `pip install --no-build-isolation .` | ✅ Works | Uses host packages including ROCm's venv wrappers |
| `pip install -e .` | ✅ Builds | But deleting source dir breaks module at runtime |
| `PREBUILD_KERNELS=2 pip install --no-build-isolation .` | ❌ Metadata gen fails | `setup.py` imports `jit` module before install completes |

### Why AITER over AOTriton on RDNA3

**Shared memory is the key constraint:**

| GPU | Architecture | Shared Memory/CU | AITER | AOTriton |
|-----|-------------|------------------|-------|----------|
| 7900XTX | RDNA3 (gfx1100) | **64 KB** | ✅ Tuned for 64 KB | ❌ Defaults to 128+ KB (CDNA) |
| MI250 | CDNA (gfx90a) | 128 KB | Works | ✅ Native target |

AITER's Triton kernels are explicitly tunable for RDNA3's 64 KB shared memory. AOTriton's pre-compiled kernels target CDNA's larger shared memory, leaving performance on the table for RDNA3.

MLA (Multi-head Latent Attention) for DeepSeek models is another AITER-exclusive advantage — the MLA decode kernel claims up to **17× speedup** over naive implementation on RDNA GPUs.

---

## Attention Backend Priority (configured in `attention.py`)

```
┌─────────────────────────────────────────────────┐
│  1. AITER flash_attn (monkey-patch SDPA)       │ ← RDNA3 optimized, 64 KB shared memory
│     Condition: arch ∈ {rdna3, rdna3_5}          │
├─────────────────────────────────────────────────┤
│  2. flash-attn Triton backend (env var)         │ ← ROCm 6.4+ / PyTorch SDPA built-in
│     Condition: flash_attn package installed      │
├─────────────────────────────────────────────────┤
│  3. AOTriton (PyTorch built-in)                │ ← Always available, CDNA-optimized
│     Version check: warn if < 0.12               │
└─────────────────────────────────────────────────┘
```

If AITER is installed AND GPU is RDNA3/RDNA3.5 (7900XTX), `configure_attention_backends()` monkey-patches `torch.nn.functional.scaled_dot_product_attention` to route through AITER's Triton kernels. Otherwise it falls through to flash-attn Triton backend or stock AOTriton.

---

## Image Tag

```
rocm-ninodes:rocm7.14-pytorch2.12.0-aotriton0.12.0-aiter0.1.13
```

All four major versioned dependencies are in the tag for reproducibility.
