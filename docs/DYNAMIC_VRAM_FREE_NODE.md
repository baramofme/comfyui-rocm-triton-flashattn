# DynamicVRAM-Free 노드 — ComfyUI dynamic VRAM에서 실제 VRAM 해제

`custom_nodes/comfyui-dynamic-vram-free/` — ComfyUI `--enable-dynamic-vram`(aimdo VBAR) 모드에서
워크플로우 전환 시 남는 VRAM을 실제로 해제하는 미니 노드.

---

## 왜 필요한가

dynamic VRAM 모드에서 ComfyUI 코어의 언로드 경로는 dynamic 모델의 VBAR(Virtual Block
Allocator) 예약을 풀지 않는다:

- `free_memory(for_dynamic=True)` → dynamic 모델은 `memory_to_free = 0`으로 evict 제외
  (`comfy/model_management.py`)
- `unload_all_models()` → `model_unload(1e30)` → `1e30 >= loaded_size`라 `partially_unload`
  (VBAR 해제) 분기를 건너뛰고 `detach()`만 실행

결과: ltx2.5(22B) → krea2 → MiniMaxH3 같은 워크플로우 전환 시 이전 모델의 VBAR이
15-20GB 남아 OOM이 발생한다. (ComfyUI issue #12335 — TextEncoder vbar cache not freed)

## 동작 원리

`ModelPatcherDynamic.partially_unload(1e32)`를 `current_loaded_models`의 모든 모델에
호출 — 이는 ComfyUI 코어가 내부적으로 VBAR을 비울 때 쓰는 정확한 경로
(`vbar.free_memory`). 표준(비-dynamic) 모델은 `partially_unload`로 CPU 오프로드.

- 모델 구조는 유지됨 — 가중치는 사용 시 파일에서 on-demand 재로드
- `soft_empty_cache(True)` + `gc.collect()` + `torch.cuda.empty_cache()` 병행
- aimdo 활성 시 `vbars_reset_watermark_limits()` + `vbars_analyze()`로 VBAR 잔여 추가 정리

## 사용법

워크플로우 전환 지점(또는 무거운 모델 로드 직전)에 노드 삽입:

1. 노드 검색: `DynamicVRAM Free (real VBAR release)`
2. 입력: 아무 연결(STRING/IMAGE/LATENT/MODEL/VAE/CLIP/CONDITIONING) — 값은 그대로 통과
   (패스스루로 워크플로우 중간에 끼울 수 있음)
3. 출력: 상태 문자열 (해제된 MB + 남은 VRAM)
4. `OUTPUT_NODE = True` — 실행 즉시 실행되고 로그에 결과 출력

## 노드 인터페이스

| 항목 | 값 |
|---|---|
| 노드 클래스 | `DynamicVRAMFree` |
| 카테고리 | `DynamicVRAM` |
| 필수 입력 | `anything` (any type) |
| 출력 | `status` (STRING) |
| 속성 | `OUTPUT_NODE`, `GLOBAL_STATE_CHANGE` |

## 설치

```bash
cp -r custom_nodes/comfyui-dynamic-vram-free /mnt/nvmedata/comfy/custom_nodes/
docker restart comfyui_gpu0
```

로드 확인: `curl -s localhost:8189/object_info | grep DynamicVRAMFree` → `True`

## 검증

- 노드 실행 후 로그: `[DynamicVRAM-Free] released XXXXMB from N model(s); VRAM now YYYYMB free`
- krea2 실행 후 VRAM 25GB 점유 상태 → 노드 실행 → free 복귀 확인

## 주의

- `current_loaded_models`가 비어 있는데 VRAM이 차 있는 경우(예: **llama-server 등
  외부 프로세스가 GPU를 점유**)에는 이 노드로 해제 불가 — 외부 프로세스의 GPU 설정을
  먼저 확인할 것 (DYNAMIC_VRAM_AMD_FINDINGS.md 8장 참고)
