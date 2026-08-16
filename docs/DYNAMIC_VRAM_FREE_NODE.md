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

---

## ⭐⭐⭐ v4 (2026-08-16): 역할 분리 — Model Fetch 노드 + Unload 노드

노드가 길어져서(입력 10 + 체크박스 10) **역할별 2노드로 분리**:

### 노드 1: `DynamicVRAM Model Fetch` (모델 수집)

| 항목 | 값 |
|---|---|
| 입력 | `model_1` ~ `model_10` (MODEL,CLIP,VAE) |
| 출력 | `models` (VRAM_MODEL_LIST — `(소켓번호, patcher)` 리스트) |
| 동작 | 연결된 모델을 patcher로 정규화해 리스트로 전달. **VRAM 안 건드림** |

### 노드 2: `DynamicVRAM Free (real VBAR release)` (언로드 실행)

| 입력 | 설명 |
|---|---|
| `anything` | 트리거/하위호환 (optional — 안 꽂아도 OUTPUT_NODE라 실행됨) |
| `models` | Fetch 노드의 리스트 입력 |
| `mode` | `unload_all` / `selective` / `show models info` |
| `use_1` ~ `use_10` | 체크박스 — **Fetch 리스트의 위치(pos)에 매핑** (소켓 번호 아님) |
| `clear_cache` | torch 캐시 정리 여부 |

- **show models info**: `(N fetched, M in registry)` + 소켓별 `in registry: yes/no` — 트리거 시점의
  실제 로드 상태 표시. 언로드 안 함.
- **unload_all**: 레지스트리 전체 언로드 + 캐시 정리 (체크박스 무시, JS 비활성).
- **selective**: 체크된 **위치**의 모델만 언로드 (내부 객체 동일성 매칭).
- 커스텀 JS: mode가 unload_all/show models info면 체크박스 비활성, selective면 활성.

### 실측 검증 (2026-08-16, krea2, fetcher←CLIP/diffusion/VAE)

| 테스트 | 결과 |
|---|---|
| selective use_1=T use_2=F use_3=T | `#1 CLIP 언로드(4999MB)`, `#2 diffusion kept`, `#3 VAE 언로드(242MB)` ✅ |
| show models info | `(3 fetched, 2 in registry)` + 소켓별 상태, VRAM 그대로 (읽기 전용) ✅ |
| 이전 selective에서 언로드된 CLIP | 다음 실행 show info에서 `in registry: no` — 상태 플래그 정확 ✅ |

### v3→v4 호환

- v2-era 워크플로우 (anything만 연결, mode 기본) → 그대로 동작 (unload_all).
- v3 워크플로우의 `model_1..10` 직접 연결은 **제거됨** — Fetch 노드로 재배선 필요.

### ⭐ 동적 제목/라벨 (v4.1, 2026-08-16) — 연결 즉시 + 실행 후 갱신

프론트엔드 JS(`web/js/dynamic_vram_free.js`)가 **연결 순간부터** 모델명을 표시:

| 시점 | Fetch 노드 | Unload 노드 |
|---|---|---|
| **연결 즉시** (onConnectionsChange) | 제목: `Fetch: [MODEL] krea2_turbo_int8_convrot…` + 소켓 라벨 = 로더의 파일명 | 체크박스 라벨 = Fetch 상류 로더 파일명 |
| **실행 후** (onExecuted) | 제목/라벨 → 클래스명+크기 (`[clip] Krea2TEModel_ 4999MB…`) | 제목 = 요약 (`selective: unloaded 2 model(s)…`), 체크박스 = 클래스명+크기 |

- 연결 시점 이름은 **상류 로더 노드의 모델명 위젯**(`unet_name`/`clip_name`/`vae_name`/
  `ckpt_name`)을 읽어 표시 — 파일명 기반. 실행 후엔 서버가 내려준 클래스명+크기로 업그레이드.
- 사용자가 제목을 직접 바꾸면 자동 갱신 중단 (`_uiAuto` 플래그).
- 검증: playwright로 실제 프론트엔드에서 연결 즉시/실행 후 갱신 모두 확인.
- 참고: 노드간 `connect()`는 (outputSlot, targetNodeId, inputSlot) 순서 — LiteGraph 0.4.
  타입 리스트(`MODEL,CLIP,VAE`)는 캔버스 드래그 연결만 허용 (raw connect는 타입 불일치 거부).

### ⭐ Set/Get 경유 연결도 갱신 (v4.2, 2026-08-16)

Fetch→Unload 사이에 **Set/Get 노드를 끼워도** 라벨이 갱신됩니다:

- **`resolveListSource`**: Unload의 models 입력 → getter 노드의 이름 키(`setnode_name`/
  `uid`/`name`/`key`/`tag`/`identifier`/`label` — 위젯 또는 입력) → 같은 키를 가진
  setter 검색 → setter의 값 입력 링크를 따라 **실제 값을 제공하는 노드(Fetch)까지 역추적**.
- **`resolveModelSource`**: 개별 모델 입력도 Set/Get + 단일 모델 입력 체인(스위치 등)을
  재귀 추적해 로더의 모델명 위젯(`unet_name`/`clip_name`/`vae_name`)까지 도달.
- 실측 (playwright): `UNETLoader→Fetch→CacheBackendData(key="models")→
  RetrieveBackendData(key="models")→Unload.models` 체인에서 체크박스 라벨이
  `krea2_turbo_int8_convrot…` 등 로더 파일명으로 즉시 갱신 확인.
- 이 설치엔 전용 Set/Get 노드가 없어 Inspire Cache/Retrieve로 검증 — 해석기는 이름 키
  기반이라 어떤 Set/Get 팩이든 동일하게 동작 (키 위젯 이름이 위 목록에 없으면 추가 필요).
- 실행 후 클래스명 갱신은 Unload 노드가 **자기 자신의** info 출력을 받아 처리하므로
  Set/Get 경유여도 정상 동작.

### ⭐ fetch 라벨 복원 (v4.3, 2026-08-16)

- v4.2에서 fetch 소켓 라벨이 `resolveModelSource` 실패 시 "model_1"로 되돌아가는 회귀 —
  **제목 폴백 복원**: `resolveModelSource(src) ?? src.title` (직접 로더→모델명,
  해석 불가 노드→소스 제목).
- `setGetKey`에 **단일 STRING 위젯 + 입력 없음** 노드를 키로 인식하는 폴백 추가 —
  이름이 알려지지 않은 Get/Set 구현도 해석 가능 (위젯 이름이 뭐든 1개짜리면 키로 취급).
- 실측 (playwright): UNETLoader→`krea2_turbo_int8_convro…`, VAELoader→`qwen_image_vae…`,
  EmptyLatentImage→제목 폴백 `Empty Latent Image`.
- 참고: ltx25 워크플로우의 GetNode/SetNode는 현재 설치에서 미등록 타입 (제공 팩이 제거됨) —
  해당 워크플로우 로드 시 Get/Set 노드가 깨져 보일 수 있음.

### ⭐⭐ 내장 Set/Get 버스 지원 (v4.4, 2026-08-16)

- **ComfyUI 내장 `SetNode`/`GetNode`** (프론트엔드 네이티브, 키 위젯 = **`Constant`**,
  서버 object_info에 없음) 식별: `setGetKey`가 **대소문자 무시** 매칭 + 단일 STRING
  위젯 폴백 (입력 유무 무관) → Set/Get 양쪽 모두 키 감지.
- Unload.models 라벨이 **Set→Get 체인을 통과**해도 표시:
  - `Fetch → Set("models") → Get("models") → Unload.models`: 체크박스에 3개 로더 이름
  - `UNETLoader → Set("unet") → Get("unet") → Unload.models`: 1번 체크박스에 모델 이름
    (리스트가 아닌 단일 모델도 `resolveModelSource` 폴백으로 1개 표시)
- 파이썬 방어: `models` 입력이 (idx,patcher) 튜플 리스트 / patcher 리스트 / 단일
  patcher 중 무엇으로 들어와도 정규화.
- 실측 (playwright, 실제 내장 SetNode/GetNode 사용): 두 시나리오 모두 라벨 갱신 확인.

### ⭐⭐⭐ reset on switch 모드 (v5, 2026-08-16) — 워크플로우 전환 시에만 전체 리셋

**목적**: 워크플로우 끝의 `unload_all`은 같은 프롬프트 재실행 시에도 모델을 전부 내려
매번 재로드 비용이 발생. **전환 시에만** 리셋하고, 같은 워크플로우 반복은 모델을
유지(속도)하고 싶을 때.

**동작**: `mode = "reset on switch"` — 첫 노드에 배치하면:
- **현재 실행 그래프의 구조 지문** 계산: 노드 class + 모델 선택 입력(`unet_name`/
  `clip_name`/`vae_name`/`ckpt_name`/`lora_name`/...) + 입력 링크 구조.
  시드/텍스트/수치(스텝·해상도)는 제외 → 같은 워크플로우의 프롬프트/시드/값 변경은
  "같은 워크플로우"로 판정.
- 직전 실행 지문(프로세스 내 전역)과 비교:
  - **다르면** → `unload_all` + 캐시 클리어 (이전 워크플로우의 모델·캐시 제거)
  - **같으면** → skip (모델 유지 — 연속 실행 속도 보존)
- 그래프 접근: `server.PromptServer.instance.prompt_queue.currently_running` —
  실행 중인 prompt dict.

**실측 (2026-08-16)**:
| 시나리오 | 결과 |
|---|---|
| 첫 실행 | `first run / workflow changed -> unloaded + cache cleared` (krea2 19.7GB → 0.74GB) |
| 같은 워크플로우 재실행 | `same workflow -> skipped` |
| 다른 그래프 | `workflow changed -> unloaded + cache cleared` |

**배치**: 워크플로우 첫 노드 뒤 (anything ← 첫 노드 출력). ltx25 _unload 파일에
`600 "RESET ALL — workflow switch only"`로 이미 추가됨 (anything ← 408 컨트롤).
- ⚠️ 시드는 지문에서 제외되지만, **모델을 바꾸면** 지문이 달라져 리셋 발동.
- 참고: 컨테이너 재시작 시 지문 초기화 → 첫 실행은 무조건 리셋 (이미 비어있어 무해).

### ⭐⭐⭐⭐ `reset_on_switch` 체크박스 (v5.1, 2026-08-16) — UX 개선

`clear_cache`처럼 **체크박스 옵션**으로 일반화:
- 입력: `reset_on_switch` (BOOLEAN, 기본 False)
- **체크하면**: `unload_all`/`selective` 어느 모드든 **새 워크플로우 첫 실행에만** 동작,
  같은 워크플로우 재실행은 skip (모델 유지 → 속도).
- **체크 안 하면**: 기존처럼 매 실행 동작.
- 기존 "reset on switch" 모드는 `unload_all + reset_on_switch=true`의 별칭 (하위 호환).
- 실측: run1 `first run / workflow changed -> unloaded + cache cleared`, run2 `same workflow -> skipped`.

### 노드 인터페이스 (v3)

| 입력 | 타입 | 설명 |
|---|---|---|
| `anything` | any (required) | 패스스루/트리거 (기존 워크플로우 하위 호환) |
| `model_1` ~ `model_10` | MODEL,CLIP,VAE | 언로드 대상 모델 연결 (선택) |
| `mode` | enum | `unload_all` / `selective` / `show models info` |
| `use_1` ~ `use_10` | BOOLEAN 체크박스 | selective에서 해당 소켓 모델 언로드 여부 |
| `clear_cache` | BOOLEAN | torch 캐시 정리 수행 여부 |

### mode 설명 (v3)

- **show models info** (구 list): 읽기 전용. 연결된 소켓별 모델 정보(클래스명/크기/장치/
  레지스트리 로드 여부) + 현재 레지스트리 목록 출력. **언로드 안 함.**
- **unload_all**: 레지스트리 전체 강제 언로드 + 캐시 정리. 체크박스 무시 (JS가 비활성화).
- **selective**: **체크된 소켓에 연결된 모델만** 언로드. 매칭은 내부 모델 객체 동일성
  (clone 공유) 기준 — 텍스트 토큰보다 정밀. 미체크 소켓·레지스트리에 없는 모델은
  건드리지 않음.

### 커스텀 JS (`web/js/dynamic_vram_free.js`)

- mode가 `unload_all` 또는 `show models info` → `use_1..use_10` 체크박스 **비활성화**
- mode가 `selective` → 체크박스 **활성화**
- 프론트엔드 새로고침 후 적용 (JS는 서버가 `/extensions/...`로 서빙)

### 모델 표시명의 한계 (소스 확인)

- ModelPatcher/CLIP/VAE/BaseModel 어디에도 `model_name` 속성 없음 (grep 0건).
- 표시는 내부 클래스명 기반: `Krea2`, `Krea2TEModel_`(텍스트 인코더), `WanVAE`,
  `CLIPModel` — aimdo 스테이징 로그와 동일 소스. 파일명(krea2_turbo_int8_convrot
  등)은 로더 노드 위젯에만 존재.

### 실측 검증 (2026-08-16, krea2 워크플로우)

| 테스트 | 결과 |
|---|---|
| selective: CLIP(체크) + diffusion(미체크) + VAE(체크) | `#1 CLIP 언로드`, `#2 diffusion kept`, `#3 VAE 언로드` — 체크박스 의미론 정확 |
| show models info | 소켓 3개 + 레지스트리 3개 보고, VRAM 19.7GB 유지 (언로드 없음) |
| unload_all (v2 검증 동일 경로) | legacy 19.7→0.74GB / dynamic 19.1→0.86GB |

### 노드 인터페이스 (v2)

| 입력 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `anything` | any (required) | - | 패스스루 (연결 필수) |
| `mode` | enum | `unload_all` | `list` / `unload_all` / `selective` |
| `model_names` | STRING | `""` | 선택 언로드 토큰 (쉼표·공백 구분) |
| `clear_cache` | BOOLEAN | `True` | torch 캐시 정리 수행 여부 |

### mode 설명

- **list**: `current_loaded_models`의 모델 목록만 출력 (언로드 안 함). 선택 언로드 전에
  어떤 이름이 있는지 확인할 때 사용.
- **unload_all**: 등록된 모든 모델 강제 언로드 + torch 캐시 정리.
- **selective**: `model_names`의 토큰 중 하나라도 모델 태그/클래스명에 매칭되면 그 모델만
  언로드. 태그 매핑: `clip` / `vae` / `vae-audio` / `vae-video` / `audio` / `video` /
  `diffusion` (클래스명 기준 — 예: `WanVAE`→vae, `Krea2`→diffusion, `CLIPModel`→clip).

### v2 동작 원리 (소스 검증, ComfyUI 0.32.0)

기존 v1의 두 가지 버그 수정:

1. **dynamic 모델**: v1은 `partially_unload(1e32)` 호출 — `device_to` 자리에 1e32가 들어가
   `memory_to_free=0` → `vbar.free_memory(0)` → **0 해제**. v2는
   `partially_unload(offload_device, 1e32)` (정확한 인자) → `vbar.free_memory(1e32)` 전체
   해제 + `partially_unload_ram(1e32)` (pinned RAM) + detach.
2. **legacy 모델**: `partially_unload(offload, 1e32)`는 **lowvram 부분로드 상태에서만**
   동작 (완전 로드 모델은 `loaded_size()==0` → 0 해제 → v1의 "released 0MB" 원인).
   v2는 `LoadedModel.model_unload(1e30)` → `detach()` 전체 언로드 + 레지스트리에서 제거.
3. **중복 해제 방지**: `_unload_one`이 `mm.current_loaded_models`에서 제거 후
   `unload_all_models()`는 unload_all 모드에서만 호출 (selective에서 나머지를 안 건드림).

### 실측 검증 (2026-08-16, RX 7900 XTX, legacy 모드 — dynamic 플래그 없음)

krea2 t2i 워크플로우 (qwen3vl CLIP + krea2 int8 UNet 13GB + WanVAE):

| 측정 (python 프로세스 드라이버 VRAM, rocm-smi) | 값 |
|---|---|
| baseline (유휴) | 0.36 GB |
| krea2 실행 후 (언로드 없음) | **19.7 GB** |
| `unload_all` 노드 후 | **0.74 GB** |
| `selective "vae"` 후 (diffusion은 등록 유지) | 로그: VAE만 언로드, VRAM 19.7→1.78 GB |

- **"released 0MB"는 이제 정확**: v1은 per-model 루프가 0을 반환해도
  `unload_all_models()`+캐시 정리가 실제 해제를 했지만 메시지에 반영 안 됨. v2는
  weights 해제량(레지스트리 기준) + torch 캐시 해제량을 분리 보고.
- **프롬프트 완료 후에는 `current_loaded_models`가 비어있음**: executor가 종료 시
  등록 해제하므로 standalone 실행은 "0 model(s)" + 캐시 정리만 수행. per-model 언로드
  경로는 **워크플로우 중간**에 노드를 둘 때 동작 (모델이 등록된 상태).
- **dynamic 모드 실측 검증 완료 (2026-08-16)**: `--enable-dynamic-vram` 추가 후 동일
  프로토콜로 측정:

  | 측정 (dynamic 모드, python 드라이버 VRAM) | 값 |
  |---|---|
  | baseline | 0.36 GB |
  | krea2 실행 후 (torch_total 0.16GB — 전부 aimdo VBAR 스테이징) | **19.1 GB** |
  | `unload_all` standalone 후 (등록 0개, 캐시 정리만) | **0.86 GB** |
  | krea2 mid-prompt (3개 dynamic 모델 등록) → `unload_all` | **0.86 GB** |

  - mid-prompt 로그: `[vae] [dynamic] WanVAE 242MB`, `[diffusion] [dynamic] Krea2
    12868MB`, `[diffusion] [dynamic] Krea2TEModel_ 4999MB` — **`[dynamic]` 태그 확인,
    `partially_unload(offload_device, 1e32)` VBAR 해제 경로 실증** (weights 합 34.8GB
    해제).
  - standalone(등록 0개) 케이스는 캐시 정리 시퀀스(soft_empty_cache + gc + empty_cache)
    가 dynamic 스테이징 버퍼까지 해제 — 기존 문서의 "Python 레벨 해제 불가" 결론은
    v2 노드에서 갱신됨. (기존 조사는 구 노드 기준이었고, 측정 시점 상태도 달랐음)

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
