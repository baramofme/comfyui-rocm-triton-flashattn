# Dynamic VRAM / MultiGPU 팩 조사 기록 — AMD(ROCm) RX 7900 XTX

ComfyUI 0.32.0 + PyTorch 2.12.0+rocm7.14.0 / RX 7900 XTX x2 (cuda:0 사용, cuda:1 유휴) 환경에서
krea2(int4) / ltx23(22B) 워크플로우의 VRAM 문제를 추적하며 **소스 레벨에서 확인한 사실들**을 정리한다.

관련 문서: [VRAM_MANAGEMENT_NODES.md](VRAM_MANAGEMENT_NODES.md) (이전 조사 — lowvram/TensorParallelV3 기준)

---

## 1. 핵심 요약 (TL;DR)

| 발견 | 내용 |
|---|---|
| **AMD에서 dynamic VRAM은 기본 꺼짐** | `main.py:249`가 `is_nvidia()` 조건으로 aimdo 초기화를 가름 → AMD는 `--enable-dynamic-vram` 명시 필수 |
| **`--lowvram`은 dynamic VRAM 활성 시 무효** | 공식 문서 + `cli_args.py:311` 확인. 단, AMD에선 dynamic이 안 켜지므로 lowvram이 여전히 유효 |
| **dynamic 모드에서 `unload_all_models()`는 VRAM을 못 놓아줌** | `model_unload(1e30)`이 VBAR 해제 분기를 우회 → detach만 하고 잔류 (krea2 15GB 잔류 확인) |
| **ComfyUI-MultiGPU는 노드를 안 써도 전역 패치를 검** | `comfy.sample.sample_custom`, `mm.get_torch_device`, `text_encoder_device`, `soft_empty_cache` 등 import 시점 패치 |
| **MultiGPU 팩 + comfy_kitchen INT8 충돌** | CUDA device 0→1 스위칭 후 DLPack export → `BufferError` |
| **ltx23 22B는 단일 GPU 24.5GB로 불가** | 텍스트 인코더 14.6GB + UNET 19.8GB = 34GB → OOM |
| **⭐ 진짜 원인: llama-server가 GPU[0]을 23.7GB 점유** | `llm-main` config에 `n-gpu-layers=99`가 활성 → GPU 오프로드 시도 후 `cudaMalloc failed: OOM`으로 실패했지만 **VRAM 23.7GB가 해제 안 된 채 점유** → ComfyUI가 같은 GPU를 못 씀. **이게 "유령 VRAM"/unload 안 됨/워크플로우 전환 OOM 전부의 실체** |

---

## 2. AMD/ROCm에서 dynamic VRAM 활성화 조건 (소스 확정)

`/workspace/main.py:249-251`:

```python
if args.enable_dynamic_vram or (enables_dynamic_vram() and comfy.model_management.is_nvidia() and not comfy.model_management.is_wsl()):
    ...
    aimdo_initialized = comfy_aimdo.control.init_devices(...)
    if aimdo_initialized:
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
        logging.info("DynamicVRAM support detected and enabled")
    else:
        logging.warning("No working comfy-aimdo install detected. DynamicVRAM support disabled. Falling back to legacy ModelPatcher...")
```

- **조건**: `--enable-dynamic-vram` OR (`enables_dynamic_vram()` AND `is_nvidia()` AND not WSL)
- **AMD에서는 `is_nvidia()` = False** → `--enable-dynamic-vram`을 **명시적으로** 줘야 aimdo가 초기화됨
- `--disable-dynamic-vram`만 지운다고 dynamic이 켜지지 않음 (실측: `enables_dynamic_vram()=True`인데도 aimdo 로그 0건, `loaded partially; lowvram patches: 0` legacy 로드 지속)

`/workspace/comfy/cli_args.py:311`:

```python
def enables_dynamic_vram():
    if args.enable_dynamic_vram:
        return True
    return not args.disable_dynamic_vram and not args.highvram and not args.gpu_only and not args.novram and not args.cpu
```

- `--lowvram`은 이 조건에 **없음** → lowvram만으론 dynamic을 막지 못함 (NVIDIA 기준)
- **AMD 실전**: dynamic이 꺼져 있으니 lowvram은 여전히 동작. ltx23 로그에 `loaded partially; ... lowvram patches: 0` 확인

### 검증 실험 기록

| 설정 | krea2 int4 | ltx23 22B |
|---|---|---|
| `--lowvram --disable-async-offload --disable-dynamic-vram` (기존) | 124초, unload "안 되는 것처럼" 보임 | 동작했으나 느림 |
| `--lowvram` (async/dynamic 플래그 제거) | **28초** (4배 빨라짐) | **BufferError** (MultiGPU 팩 개입) |
| MultiGPU 팩 비활성화 + 위 플래그 | (미확인) | **OOM** — 34GB > 24.5GB |

---

## 3. dynamic 모드에서 `unload_all_models()`가 VRAM을 못 놓는 이유 (소스 확정)

`/workspace/comfy/model_management.py:797`:

```python
def model_unload(self, memory_to_free=None, unpatch_weights=True):
    if memory_to_free is not None:
        if memory_to_free < self.model.loaded_size():
            freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
            if freed >= memory_to_free:
                return False
    self.model.detach(unpatch_weights)
    ...
    return True
```

- `unload_all_models()` → `free_memory(1e30, dev)` → `model_unload(1e30)`
- **`1e30 >= loaded_size` (12.9GB)가 항상 True** → `partially_unload()`(VBAR 해제 분기)를 **건너뜀**
- `detach()`만 실행 → **dynamic 모델의 VBAR(aimdo Virtual Block Allocator) 예약이 프로세스에 잔류**
- 결과: `/free` API, MCP clear_vram, UnloadAllModels 노드 — **모두 같은 경로라 전부 안 풀림**

### dynamic 모델의 VRAM 관리 구조

`/workspace/comfy/model_patcher.py:1791+` (ModelPatcherDynamic):

```python
def is_dynamic(self): return True

def _vbar_get(self, create=False):
    ...
    vbar = comfy_aimdo.model_vbar.ModelVBAR(self.model_size() * 10, self.load_device.index)  # 10배 가상 공간
    ...

def partially_unload(self, device_to, memory_to_free=0, ...):
    vbar = self._vbar_get()
    vbar_freed = 0 if vbar is None else vbar.free_memory(memory_to_free)  # ← 유일한 해제 경로
    ...
```

- dynamic 모델은 `comfy_aimdo.model_vbar.ModelVBAR`로 메모리 관리
- **VBAR 해제는 `partially_unload()` 경유뿐인데, `unload_all_models()`는 그걸 안 부름**
- `pin_weight_to_device`/`unpin_weight`은 `RuntimeError` — dynamic 모델에선 pin API 불가

### 관측된 증상 (krea2, dynamic 모드)

- `Requested to load WanVAE... 242.03 MB loaded` — 로드된 모델은 VAE 242MB뿐
- 그런데 VRAM 15.1GB 잔류 (torch 예약은 432MB, KFD 프로세스 할당은 GPU[1]에 776MB뿐)
- → **"유령 예약"의 정체 = VBAR 잔류**. 노드/MCP/API가 전부 같은 경로라 못 푸는 것

---

## 4. ComfyUI-MultiGPU 팩 — "노드를 안 써도" 전역 패치

**결론: 이 팩은 import 시점에 전역 패치를 걸어서, MultiGPU 노드를 하나도 안 쓰는 워크플로우까지 망가뜨린다.**

### 패치 목록 (실측 로그)

| 패치 | 위치 | 영향 |
|---|---|---|
| `comfy.sample.sample` / `sample_custom` | `__init__.py:513-524` | 샘플링마다 모델 load_device로 CUDA 전환 시도 |
| `mm.get_torch_device` / `text_encoder_device` / `unet_offload_device` | `__init__.py:582` | device 선택 가로챔 → `text_encoder_device_patched returning device: cpu` |
| `mm.soft_empty_cache` | `device_utils.py:232` | VRAM 정리 대체 + "PromptExecutor cache reset" 트리거 |
| `comfy.sd.load_state_dict_guess_config` | `checkpoint_multigpu.py:29` | config hash가 없으면 원본 호출 (무해하지만 개입) |
| `ModelPatcher.partially_load` | `distorch_2.py:332` | distorch meta 없는 모델은 원본 호출 |
| `comfy_kitchen DLPack device guard` | `__init__.py` | comfy_kitchen HIP INT8과 상호작용 |

### 관측된 피해 (MultiGPU 노드 없는 워크플로우에서)

- **krea2**: `[MultiGPU Core Patching] text_encoder_device_patched returning device: cpu` → 텍스트 인코더가 CPU로 강제 (lowvram과 혼동)
- **ltx23**: `[MultiGPU CUDA Guard] Switching CUDA current device 0 -> 1 (NodeOverride.load_unet)` → UNET이 GPU[1]로 감

### comfy_kitchen DLPack BufferError (ltx23 실패 지점)

```
BufferError: Can't export tensors on a different CUDA device index. Expected: 0. Current device: 1.
  File "comfy_kitchen/backends/hip/__init__.py", line 657, in int8_linear
    _dl(q), _dl(weight), _dl(out),
  File "comfy_kitchen/backends/hip/__init__.py", line 265, in _dl
    return t.__dlpack__(stream=-1)
```

- MultiGPU CUDA Guard가 device를 0→1로 바꾼 뒤, **GPU[0] 연산 중 GPU[1] 텐서를 DLPack export** 시도 → 실패
- comfy_kitchen HIP 백엔드의 `__dlpack__`은 단일 GPU 전제

### 비활성화 방법 (되돌리기 쉬움)

```bash
cd /mnt/nvmedata/comfy/custom_nodes
mv ComfyUI-MultiGPU ComfyUI-MultiGPU.disabled
mv ComfyUI-DistorchMemoryManager ComfyUI-DistorchMemoryManager.disabled
# docker restart comfyui_gpu0
```

비활성화 후: `MultiGPU Core Patching` 로그 0건, GPU[0] VRAM 387MB로 정상 복귀.

---

## 5. ltx23 22B — MultiGPU 팩의 양날 (핵심 트레이드오프)

| 상태 | 결과 |
|---|---|
| **MultiGPU 팩 ON** | GPU[1]로 분산 → comfy_kitchen DLPack BufferError (device 0↔1) |
| **MultiGPU 팩 OFF** | 단일 GPU 24.5GB → 텍스트 인코더 14.6GB + UNET 19.8GB = 34GB → **OOM** |

- 로드 순서 (팩 OFF, 로그): VideoVAE 1.4GB → LTXAVTEModel_ 14.6GB full → LTXAV 19.8GB partial → AudioVAE 0.7GB → lowvram 부분 언로드 반복 → `Got an OOM, unloading all loaded models` (388초)
- **이 워크플로우(ltx23ErosSeedHunter_v2.json)는 원래 22B라 두 GPU 분산이 필요** — 팩을 끄면 용량 자체가 안 됨
- `aimdo_enabled = False` (MultiGPU 팩이 유일하게 aimdo를 켜던 경로였음) → dynamic 스트리밍(VBAR)도 불가 → legacy full load 폴백

### 열린 질문

1. **`--enable-dynamic-vram` 명시** → aimdo ON → VBAR 스트리밍으로 34GB를 24.5GB + 파일 스트리밍으로 처리 가능한지 (미검증)
2. `--enable-dynamic-vram` + MultiGPU 팩 OFF 조합이 ltx23을 돌릴 수 있는지
3. dynamic ON 시 krea2의 `unload_all` 미동작(3장)과의 트레이드오프

---

## 6. flash attention 경고 — 무해

```
[WARNING] Flash Attention failed, using default SDPA: Mask must not be set for Flash attention
```

- `comfy/ldm/modules/attention.py:842` — `attention_flash`가 mask 있는 입력에서 **의도적으로** raise → SDPA 폴백
- krea2 edit의 `ref_boost=4` 마스크 어텐션 경로에서 발생. **품질 영향 없음, 속도만 미미하게 저하**
- 어텐션 선택 순서 (소스): sage > flash > xformers > pytorch(SDPA) > split/quad
- SDPA(pytorch attention)는 flash보다 **느림** — "더 빠르다"는 오해. mask가 있을 때의 폴백일 뿐

---

## 7. MIOpen 첫 실행 지연 (VAE 디코드 20분 멈춤 현상)

### 현상

- `--enable-dynamic-vram` 전환 후 첫 ltx23 실행: 샘플링은 46초로 정상인데 **VideoVAE 디코드에서 ~20분 정체** (GPU 98%로 도는데 진행 로그 없음 → 멈춘 것처럼 보임)
- 로그에 `MIOpen(HIP): Warning [IsEnoughWorkspace] [EvaluateInvokers] Solver <GemmFwdRest>, workspace required: 676177920, provided ptr: 0 size: 0` 수백 건 폭주
- 완료 후 `~/.cache/miopen/3.5.2.cd957402/` + `~/.config/miopen/gfx1100_48.*.udb.txt/ufdb.txt` 생성 → **이후 실행은 정상 속도 (MIOpen 경고 0건, 디코드 수 초)**

### 원인 (웹 조사로 확정 — GitHub 이슈/공식 문서)

1. **cold MIOpen cache에서 커널 검색이 수십 분** — PyTorch PR #179795 (ROCm CI): ConvTranspose3d cold cache 기준
   | Find mode | 소요 |
   |---|---|
   | 기본값 | ~13.2분 |
   | `MIOPEN_FIND_MODE=FAST` | ~1.2분 |
   
   > "On ROCm with a cold MIOpen cache, each op benchmark config can take 10–15 min due to MIOpen's kernel search"

2. **`IsEnoughWorkspace ... provided ptr: 0 size: 0`** (ROCm/MIOpen #2981, rocm-libraries #4071): PyTorch가 conv find 호출 시 **workspace 포인터를 전달하지 않아서** MIOpen이 각 solver를 "이 workspace로 실행 가능?" 평가하며 경고. **경고 자체는 무해하나 결과가 sub-optimal** (성능 저하 동반). `MIOPEN_FIND_MODE=FAST` 설정 시 경고 사라짐 (우리 서버가 이미 FAST → 2번째 실행부터 0건)

3. **캐시 구조** (MIOpen 공식 docs):
   - `~/.cache/miopen/*/` → **커널 바이너리 캐시 (ukdb)**
   - `~/.config/miopen/*.ufdb.txt` → **find 결과 DB (사용자 DB)**
   - 첫 실행에 채워지고 이후 재사용 → 빨라짐

4. **⚠️ 캐시가 항상 재사용되는 건 아님**:
   - rocm-libraries #3553: find-db가 프로세스 간 재사용 안 되는 버그 (InvokerCache가 메모리 전용 → 매 프로세스마다 레이어당 15-16초 재벤치)
   - ROCm #6008 (ComfyUI WAN VAE 사례 — 우리와 유사): `feat_cache`(temporal cache) 때문에 **Run 1과 Run 2의 텐서 shape이 달라져서** 새 shape마다 재벤치
   - 확인법: `MIOPEN_LOG_LEVEL=5` 로그에서 `FindSolutionImpl` 검색 (재검색 여부)

### ⚠️ 캐시 휘발성 (우리 환경)

- 캐시 위치: 컨테이너 내부 `/root/.cache/miopen/` + `/root/.config/miopen/`
- **docker-compose 볼륨 마운트에 없음** (hf-hub/torch-hub만 마운트됨) → **컨테이너 재생성 시 캐시 소실 → 다음 첫 실행에서 또 20분 지연**
- **해결: 볼륨 마운트 추가**
  ```yaml
  volumes:
    - /mnt/nvmedata/comfy/miopen-cache:/root/.cache/miopen
    - /mnt/nvmedata/comfy/miopen-config:/root/.config/miopen
  ```

### 워밍업 노드 vs 캐시 영속화 (판단)

- **워밍업 노드는 비권장**: (1) MIOpen 캐시가 프로세스/컨테이너 수명에 묶여 있어 재생성 시 의미 없음, (2) 임의 shape으로 워밍업하면 실제 shape과 안 맞아 캐시 미스 (feat_cache로 run마다 shape 변하는 것도 확인), (3) "첫 실행의 지연을 다른 시점으로 옮길" 뿐
- **캐시 영속화가 정답**: 한 번 만들어진 캐시가 컨테이너 재생성에도 유지. 추가 노드 0개, docker-compose 볼륨 2줄만
- 새 해상도/프레임 shape이 처음 나올 때는 어차피 그 shape이 실제 나타날 때 캐시가 채워짐 (일회성 비용)

### 현재 권장 환경변수

```
MIOPEN_FIND_ENFORCE=1       # 캐시 강제 사용
MIOPEN_FIND_MODE=FAST       # 휴리스틱만 (경고 억제 + 검색 최소화) — 유지 권장
COMFYUI_ENABLE_MIOpen=1
```

---

## 8. ⭐ 최종 결론 — 진짜 원인은 llama-server의 VRAM 점유

### 모든 VRAM 미스터리의 실체

지금까지의 "유령 VRAM", "unload_all_models()가 안 되는 것처럼 보임", "재시작해도 VRAM이 안 풀림",
"워크플로우 전환 시 OOM" — 전부 **하나의 원인**으로 설명된다:

**`llm-main` 컨테이너의 llama-server가 GPU[0]에 23.7GB를 점유하고 있었다.**

### 발단 (증거)

`/opt/llm/llama-cpp/main-llm-config.ini`의 전역 `[*]` 섹션:

```ini
[*]
n-gpu-layers = 99   # ← 활성화되어 있었음 (사용자는 CPU 전용으로 알고 있었음)
```

- llama-server 실행 인자: `--n-gpu-layers 99` → **GPU 오프로드 시도**
- 로그: `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 13573.92 MiB on device 0: cudaMalloc failed: out of memory`
  → GPU 할당 실패 후 **CPU fallback으로 추론** (사용자가 본 "CPU로 돌고 있다"는 맞음)
- **하지만 실패 전/후 할당된 VRAM 23.7GB가 해제되지 않고 점유** — KFD 프로세스 목록으로 확정:
  ```
  PID 543225  llama-server  GPU[0]  23666753536  (23.7GB)
  ```
- GPU 사용률은 0% (실제 계산은 CPU) → "GPU 안 쓰는데 VRAM만 잡혀있는" 상태

### 왜 "재시작해도 안 풀렸나"

- ComfyUI를 아무리 재시작해도 **llama-server는 계속 살아있어서** VRAM 23.7GB가 유지됨
- `current_loaded_models: 0` + aimdo `devctxs: []` + torch `reserved 0.4GB` — **ComfyUI/aimdo/torch 전부 비어있는데 VRAM만 차지** → "유령"처럼 보임
- KFD에 프로세스가 보이지 않아서(컨테이너 PID 네임스페이스) 더 헷갈림

### GPU 인덱스 함정 (진단 시 주의)

- `rocm-smi`의 `GPU[0]` = drm `card2` (unique `0x4792...`)
- `rocm-smi`의 `GPU[1]` = drm `card0` (unique `0x9d8f...`)
- **인덱스가 반대** — `rocm-smi --showpids`의 GPU 번호와 `/sys/class/drm/card*`를 cross-check 필수

### 해결 (적용 완료)

```bash
# 호스트에서 config 수정 (컨테이너는 /app/config.ini로 마운트됨)
sed -i 's/^n-gpu-layers = 99$/n-gpu-layers = 0/' /opt/llm/llama-cpp/main-llm-config.ini
docker restart llm-main
```

결과:

| | 이전 (문제) | 지금 (해결) |
|---|---|---|
| llama-server GPU 오프로드 | `n-gpu-layers 99` → GPU[0] 23.7GB | `n-gpu-layers 0` → CPU 전용, GPU 0 |
| ComfyUI cuda:0 가용 VRAM | 0.4GB | **24.8GB** |
| 워크플로우 전환 OOM | 발생 | **해결** |

검증: `rocm-smi --showpids`에서 llama-server `VRAM USED 0` 확인, ComfyUI `system_stats` cuda:0 free=24.8GB.

### 교훈

1. **"GPU 안 쓴다" ≠ "VRAM 안 잡는다"** — CPU 추론이어도 오프로드 설정이 켜져 있으면 VRAM을 점유할 수 있음
2. VRAM 문제 진단 시 **컨테이너 밖 프로세스(특히 다른 GPU 앱)를 먼저 확인** — KFD 프로세스 목록 + 카드별 unique_id 매핑
3. `rocm-smi` GPU 인덱스와 drm card 인덱스는 반대일 수 있음
4. 이후 작업: `ComfyUI-MultiGPU.disabled`/`ComfyUI-DistorchMemoryManager.disabled`는 필요 시 원복 가능. `comfyui-dynamic-vram-free` 노드는 VRAM이 넉넉해져서 불필요해졌음 (삭제 가능)

---

## 9. 참고 — 이번에 사용한 진단 경로

- 로그: `docker logs comfyui_gpu0` (핵심: `Requested to load`, `loaded partially/completely`, `Prompt executed`)
- VRAM: `rocm-smi --showmeminfo vram` (KFD: `rocm-smi --showpids`)
- 시스템: `http://localhost:8189/system_stats` (torch 예약 vs 드라이버 사용 구분 가능)
- 실패 진단: `comfyui_get_history diagnose` (정확한 실패 노드 + traceback)
- 행 판별: py-spy `dump --pid 1` (멈춤 vs 진행 중 구분 — GPU 98% + `voluntary_ctxt_switches` 증가면 진행)
- 핵심 소스: `/workspace/main.py:249`, `/workspace/comfy/model_management.py:797`, `/workspace/comfy/model_patcher.py:1791+`
- 플래그 문서: https://docs.comfy.org/development/comfyui-server/startup-flags
- MIOpen 참고: PyTorch PR #179795, ROCm/MIOpen #2981, rocm-libraries #3553/#4071, ROCm #6008
