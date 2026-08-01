# VRAM 관리 노드 가이드 — Krea2 (이미지) vs LTX 2.3 (영상)

RX 7900 XTX 24GB 단일 GPU 기준. 조사 결과를 바탕으로 각 워크플로우에 적용 가능한 노드와 lowvram 조합을 정리한다.

---

## 1. 핵심 전제 (왜 LTX만 특별한가)

| | Krea2 (이미지) | LTX 2.3 (영상) |
|---|---|---|
| 모델 크기 | int8 13GB | int8 21GB (24GB의 87%) |
| 여유 VRAM | ~11GB (활성화 공간 충분) | ~2-3GB (활성화 공간 거의 없음) |
| OOM 원인 | 고해상도 시에만 어텐션 | 시퀀스 27,280토큰의 FFN/어텐션 활성화 |
| lowvram 효과 | 있음 (CLIP 5GB → CPU, 20.7s 유지) | 거의 없음 (텍스트 인코더만 CPU로 빼도 모델이 이미 21GB) |

**결론: Krea2는 lowvram으로 충분. LTX는 lowvram이 아니라 "청킹"이 답.**

---

## 2. LTX 2.3 영상 — 적용 중인 노드 (권장)

### 2.1 Tensor Parallel V3 (Safe FFN Chunking) ⭐ 핵심

- **노드**: `TensorParallelV3Node` (`comfyui_tensor_parallel_v3`, ComfyUI_LTX-2_VRAM_Memory_Management 하위 폴더)
- **설치**: 하위 폴더 `comfyui_tensor_parallel_v3/`를 `custom_nodes/` 루트로 이동 (README 지침)
- **설정**: `ffn_chunks=8` (기본), `verbose=1`
- **동작**: 모델 내 모든 FFN(`block.ff` + `block.audio_ff` 112개)의 forward를 래핑. seq_len을 8조각으로 나눠 순차 계산 → FFN 활성화 peak 1/8.
- **검증**: 로그에 `Wrapped 112 FFN modules` 출력 확인. 24GB 단일 GPU에서 OOM 없이 성공.
- **주의**: 모델 구조가 `.net` + `.ff` 이름인 모델 전용 (LTX OSS 구조). Krea2(Flux 계열, `mlp`/`SwiGLU`)에는 **0개 래핑** — 적용 안 됨.

### 2.2 MuseDirectorSamplerV1 시간 축 chunk

- **노드**: `MuseDirectorSamplerV1` (`muse-ltx-timeline`)
- **설정**: `chunk_duration_seconds=10` (12초 → 10s+2s), `carry_frames` 자동
- **동작**: 시간 축으로 시퀀스를 분할 → 어텐션 시퀀스가 T 단위로 반토막. **V3와 직교** — V3는 FFN(메모리 축), 이건 시간 축.
- **검증**: single-pass(1 chunk)는 OOM, chunked(2 chunks)는 성공 (muse_00041_000.mp4).
- OOM이 나면 `chunk_duration_seconds`를 더 줄이기 (10 → 5~6).

### 2.3 해상도 축소 (노드 아님, 워크플로우 레벨)

- 1280×704, 288프레임에서 성공. 해상도/프레임을 올리면 2.1/2.2 조정 필요.

---

## 3. Krea2 이미지 — 적용 중인 설정 (권장)

### 3.1 `--lowvram` CLI (필수)

- **CLI**: `--disable-pinned-memory --enable-manager --enable-manager-legacy-ui --disable-mmap --disable-api-nodes --disable-dynamic-vram --disable-async-offload --use-flash-attention --lowvram`
- **동작**: CLIP/텍스트 인코더만 CPU로 오프로드, UNet(디퓨전 모델)은 GPU 상주. reserve-vram(UNet thrash)과 다름.
- **검증**: 연속 4회 20.70/20.66/20.73/20.67s, OOM 없음. reserve-vram은 3분+로 역효과였음.

### 3.2 `--disable-dynamic-vram` (필수, AMD)

- main.py 57행 `comfy_aimdo.control.init()`이 `is_nvidia()` 조건 없이 실행됨 → disable 없으면 AMD에서도 aimdo prefetch가 켜져 검은 이미지 재발.

### 3.3 고해상도 OOM 시에만 (현재 불필요)

| 우선순위 | 노드 | 대상 |
|---|---|---|
| 1 | `VAEDecodeTiled` / `VAEEncodeTiled` (내장) | VAE 메모리 |
| 2 | `PatchModelAddDownscale` (Kohya Deep Shrink) | 어텐션 시퀀스 (block 3, 0~35% 구간) |
| 3 | `ComfyUI-TiledDiffusion` | 디퓨전 자체 타일링 (2K+) |

**Krea2에 TensorParallelV3 불가** — SwiGLU가 `.net` 속성 없음, 이름이 `.mlp`. 0개 래핑으로 무효.

---

## 4. 조합 매트릭스

| 조합 | Krea2 | LTX | 비고 |
|---|---|---|---|
| `--lowvram` | ✅ 필수 | ❌ 무의미 | LTX는 모델이 이미 21GB |
| `--disable-dynamic-vram` | ✅ 필수 | ✅ 필수 | AMD 검은 이미지 방지 |
| TensorParallelV3 (ffn_chunks=8) | ❌ 구조 불일치 | ✅ 필수 | 112개 FFN 래핑 |
| MuseDirector 시간 chunk | — | ✅ 필수 | V3와 직교 |
| PatchModelAddDownscale | ⚠️ 고해상도 시 | ❌ RoPE 충돌 | LTX는 freqs 1회 계산 → 다운스케일 불가 |
| TiledDiffusion | ⚠️ 2K+ | ❌ 4D 전제 | 영상 5D latent에서 깨짐 |
| reserve-vram | ❌ 역효과 | ❌ | UNet CPU 왕복 3분+ |
| GC 0.5 | ❌ | ❌ | OOM 미해결, 0.8 유지 |

---

## 5. 모델별 적용 시 트레이드오프 요약

- **LTX 22B**: 가중치 21GB가 24GB의 87%. 활성화 공간이 2-3GB뿐이라 **활성화를 줄이는 모든 기법(V3, 시간 chunk)이 정답**. Deep Shrink처럼 해상도를 바꾸는 기법은 RoPE(프리퀀시 1회 계산)와 충돌해 사용 불가.
- **Krea2**: 가중치 13GB로 여유 있음. lowvram이면 충분하고, 고해상도로 갈 때만 어텐션 방향 기법(VAE Tiled → Deep Shrink → TiledDiffusion) 순서로 추가.

---

## 6. 참고

- 검증 로그: `docker logs comfyui_gpu0` (TensorParallelV3: `Wrapped 112 FFN modules`)
- V3 설치 원본: `/workspace/custom_nodes/ComfyUI_LTX-2_VRAM_Memory_Management/comfyui_tensor_parallel_v3/`
- LTX 모델 선택: fp8 미보유. int8(21G)은 단일 GPU 불가, int4(17G)⚠️ / UD-Q4_GGUF(16G)✅
