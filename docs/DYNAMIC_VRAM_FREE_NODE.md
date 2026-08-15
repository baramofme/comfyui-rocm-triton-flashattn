# DynamicVRAM-Free 노드 — ComfyUI dynamic VRAM에서 VRAM 해제 (및 한계 기록)

`custom_nodes/comfyui-dynamic-vram-free/` — ComfyUI `--enable-dynamic-vram`(aimdo VBAR) 모드에서
`current_loaded_models`에 남은 모델의 VBAR 예약을 해제하는 미니 노드.

---

## 왜 필요한가

dynamic VRAM 모드에서 ComfyUI 코어의 언로드 경로는 dynamic 모델의 VBAR(Virtual Block
Allocator) 예약을 풀지 않는다:

- `free_memory(for_dynamic=True)` → dynamic 모델은 `memory_to_free = 0`으로 evict 제외
  (`comfy/model_management.py`)
- `unload_all_models()` → `model_unload(1e30)` → `1e30 >= loaded_size`라 `partially_unload`
  (VBAR 해제) 분기를 건너뛰고 `detach()`만 실행

결과: ltx2.5(22B) → krea2 → MiniMaxH3 같은 워크플로우 전환 시 이전 모델의 VBAR이
15-20GB 남아 OOM이 발생할 수 있다. (ComfyUI issue #12335 — TextEncoder vbar cache not freed)

## 동작 원리

`ModelPatcherDynamic.partially_unload(1e32)`를 `current_loaded_models`의 모든 모델에
호출 — 이는 ComfyUI 코어가 내부적으로 VBAR을 비울 때 쓰는 경로 (`vbar.free_memory`).
표준(비-dynamic) 모델은 `partially_unload`로 CPU 오프로드.

- 모델 구조는 유지됨 — 가중치는 사용 시 파일에서 on-demand 재로드
- `soft_empty_cache(True)` + `gc.collect()` + `torch.cuda.empty_cache()` 병행

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
- `current_loaded_models`에 모델이 있는 상태에서만 해제 동작 (krea2 실행 직후 등)

---

## ⚠️ 실제 조사에서 밝혀진 한계 (중요)

### 1. VRAM 99% 케이스에서는 해제 불가

krea2 실행 후 VRAM 99%(17.8GB 사용) 상태에서 이 노드를 실행해도 **거의 안 풀린다**
(52MB만 해제). 원인:

```
current_loaded_models: 0      ← ComfyUI "모델 없음"
aimdo get_total_vram_usage: 0 ← aimdo "VBAR 없음"
torch reserved: 0.00GB        ← torch "비어있음"
드라이버 사용량: 17.8GB        ← 실제
```

**잔류 VRAM은 dynamic VRAM이 "다음 실행을 위해 캐시해 둔" 스테이징 버퍼**이며,
torch/aimdo/current_loaded_models 어디에도 노출되지 않는 HIP 직접 예약이다.
Python 레벨에서 접근 불가 → 이 노드로 해제 불가.

### 2. `deinit()`은 VRAM을 풀지만 프로세스를 크래시시킴 (시도 → 롤백)

`comfy_aimdo.control.deinit()` + `init_devices()` 재초기화 방식은 **12GB 해제에 성공**했지만
C 라이브러리 언로드로 ComfyUI 프로세스가 세그폴트로 죽었다:

```
AttributeError: 'NoneType' object has no attribute 'get_devctx'
model_vbar.py:132 in __del__  ← 크래시 지점
```

**결론: `deinit()`은 사용 불가. VRAM을 풀어도 서버가 죽으면 의미가 없다.**

### 3. 동적 VRAM 캐시는 프로세스 수명에 묶여 있음

- 17.8GB 잔류는 **의도된 캐시** — 같은 워크플로우 연속 실행 시 재사용되어 오히려 빨라짐
  (krea2 65.7s → 57.9s 확인)
- **다른 워크플로우 전환 시에만 문제** — 새 모델 스테이징 공간이 부족해 지연
- Python 레벨에서 안전하게 해제할 방법이 없으므로, **프로세스 재시작이 유일한 해제 수단**

### 권장 운영 방식

| 상황 | 방법 |
|---|---|
| 같은 워크플로우 반복 | 그냥 실행 (캐시 활용, 빨라짐) |
| 다른 워크플로우 전환 | `docker restart comfyui_gpu0` (~30초) — 유일한 안전한 해제 |
| 전환 직후 OOM/지연 | 위 재시작으로 해결 |
| 이 노드 | `current_loaded_models`에 모델이 남아있는 좁은 케이스에서만 유효 (krea2 직후 등) |

## 주의

- `current_loaded_models`가 비어 있는데 VRAM이 차 있는 경우에는 이 노드로 해제 불가:
  - **동적 VRAM 스테이징 캐시** (프로세스 재시작 필요)
  - **llama-server 등 외부 프로세스가 GPU를 점유** (외부 프로세스 설정 확인 —
    DYNAMIC_VRAM_AMD_FINDINGS.md 8장 참고)
- `deinit()`/`init_devices()` 재초기화 시도는 **프로세스 크래시**를 유발하므로 절대 금지
