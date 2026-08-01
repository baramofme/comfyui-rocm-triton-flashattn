# Black Images from 2nd Generation — Debug Log

> Debug record in git-commit-message style: **현상 (Symptom) / 추측&시도 (Hypothesis & Attempts) / 결과 (Result)**
> Resolution date: 2026-08-01 · Stack: ROCm 7.14 / torch 2.12.0+rocm7.14 / ComfyUI v0.28.2 / RX 7900 XTX (gfx1100)

---

## fix: black image from 2nd generation on AMD — disable dynamic-vram (comfy-aimdo)

### 현상 (Symptom)

- **1회차 생성**: 정상적으로 노이즈 제거된 이미지
- **2회차부터**: 검은색 이미지 (또는 노이즈 미제거) 반복
- 컨테이너 재시작하면 다시 1회차만 정상 → **상태성(stateful) 버그**
- 생성 로그에 `RuntimeWarning: invalid value encountered in cast` 가 **2회차 블록에만** 등장
  - 출처: `nodes.py:1682` (SaveImage) — `np.clip(i, 0, 255).astype(np.uint8)` 경로에서 NaN 검출
  - 1회차에는 경고 없음 → **2회차 출력에 NaN 확정**

### 추측&시도 (Hypothesis & Attempts)

| # | 추측 | 시도 | 결과 |
|---|------|------|------|
| 1 | 최신 Triton/AITER 어텐션 커널이 SDPA 대신 활성화되어 오작동 | `FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE`, `AITER_TRITON_ONLY=0`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0` | ❌ 현상 유지 |
| 2 | TunableOp 캐시가 1회차 커널 튜닝 결과를 잘못 재사용 | `PYTORCH_TUNABLEOP_ENABLED=0` + `rm -rf /cache/tunableop` | ❌ 현상 유지 |
| 3 | MIOpen find-db 캐시 재사용 문제 | `MIOPEN_FIND_ENFORCE=0`, `MIOPEN_DEBUG_DISABLE_FIND_DB=1` + `rm -rf /cache/miopen` | ❌ 현상 유지 |
| 4 | async weight offload 스트림 재사용 오염 | `--disable-async-offload` (AMD 기본 2-stream off) | ❌ 현상 유지 |
| 5 | **comfy-aimdo DynamicVRAM prefetch 시스템** | **`--disable-dynamic-vram`** | ✅ **해결** |

### 결과 (Result)

- `--disable-dynamic-vram` 적용 후 **3회 연속 생성 전부 정상** (32.9s / 27.3s / 27.2s)
- `RuntimeWarning: invalid value encountered in cast` **전무** — NaN 완전 소멸
- 부팅 로그의 prefetch 메시지(`12864MB Staged`, `Force pre-loaded`, `prepared for dynamic VRAM loading`) 사라지고 `loaded completely; ... full load: True` 로 대체

### 근본 원인 (Root Cause)

`comfy-aimdo 0.4.10`의 DynamicVRAM prefetch 경로(`ModelPatcherDynamic` + `comfy/ops.py` vbar 경로)가 FP8 모델 웨이트를 in-place로 변형:

```python
# comfy/ops.py — resolve_cast_module_with_vbar() → post_cast()
if (want_requant and len(fns) == 0 or update_weight):
    seed = comfy.utils.string_to_seed(s.seed_key)
    if isinstance(orig, QuantizedTensor):
        y = orig.requantize_from_float(x, scale="recalculate", stochastic_rounding=seed)
if update_weight:
    orig.copy_(y)   # ← 1회차 추론 중 원본 웨이트 in-place 재양자화

# comfy/ops.py:953 — "# Cache FP8 quantized input"  ← 오염된 캐시 재사용
```

1회차 추론에서 FP8 웨이트가 재양자화로 in-place 변형되고, 2회차부터 오염된 캐시/웨이트를 재사용 → NaN → 검은 이미지.
`--disable-async-offload`만으로 무효였던 이유: aimdo prefetch가 그대로 활성 상태였기 때문.

### 적용된 설정 (docker-compose.yml)

```yaml
environment:
  - CLI_ARGS=--disable-pinned-memory --enable-manager --disable-dynamic-vram --disable-async-offload
```

### 관련 파일

- `comfy/ops.py`: 213 `resolve_cast_module_with_vbar`, 258-267 `requantize_from_float` / `orig.copy_(y)`, 833 `cast_bias_weight(..., want_requant=True)`, 953 `# Cache FP8 quantized input`
- `comfy/model_patcher.py`: 1864 `m.seed_key = n` (웨이트 키 기반 결정적 시드)
- `comfy/model_management.py`: 1290-1301 async offload 기본값(AMD 2-stream)
- `nodes.py`: 1675-1690 SaveImage NaN 캐스트 경고 지점
- 워크플로: `krea2_turbo_fp8_scaled.safetensors` + `qwen3vl_4b_fp8_scaled.safetensors` (FP8 양자화 모델)

### 남은 과제 (Open Items)

- comfy-aimdo 쪽 근본 버그 리포트 (vbar prefetch의 in-place requantize)
- ~~fa2(flash-attn 2.8.3)는 `flash_attn_2_cuda` 네이티브 확장 미빌드 상태로 설치 — 실사용 불가 확인~~ → **해결**: `--use-flash-attention` 플래그로 FA2 Triton 경로 정상 동작 확인 (벤치 20.54s, SDPA 대비 -6.8%)
- ~~aiter 0.1.13은 ComfyUI 연동 없음 (dormant)~~ → **해결**: aiter는 FA2의 Triton 백엔드로 `--use-flash-attention` 경유해 사용 (커스텀 노드 불필요, AMD 공식 블로그 확인)
- ~~어텐션 성능: SDPA FLASH 15ms vs EFFICIENT 0.3ms — EFFICIENT 강제 검토~~ → **종결**: 실워크플로 벤치에서 무차별 (성능 문서 참고)
