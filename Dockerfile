FROM rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0

# ponytail: ROCm SDK Core lib path for aiter JIT build (linker needs libamdhip64.so, runtime needs libamdhip64.so.7)
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib

RUN ln -sf /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7 /opt/venv/lib/libamdhip64.so

ENV MIOPEN_FIND_ENFORCE=1 \
    MIOPEN_FIND_MODE=FAST \
    MIOPEN_DEBUG_DISABLE_FIND_DB=0 \
    PYTORCH_TUNABLEOP_CACHE_DIR=/cache/tunableop \
    TRITON_CACHE_DIR=/cache/triton \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    COMFYUI_ENABLE_MIOpen=1 \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8 \
    MALLOC_MMAP_THRESHOLD_=65536 \
    MALLOC_TRIM_THRESHOLD_=65536

# ponytail: AITER_TRITON_ONLY=1 (triton-only AITER kernels) breaks gfx1100 — black images from 2nd generation onward
ENV AITER_TRITON_ONLY=0

# ponytail: ROCr 1.21 AsyncEventsLoop busy-spin fix (ROCm/TheRock#7051) —
# preload clean ROCr 1.18 + 10-symbol shim over the bundled _rocm_sdk_core runtime
COPY docker/rocr-fix/ /opt/rocr-fix/
ENV LD_PRELOAD=/opt/rocr-fix/libhsa_shim.so:/opt/rocr-fix/libhsa-runtime64.so.1.18.70203

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    nano \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ARG COMFYUI_VERSION=v0.32.0
RUN git clone --depth 1 --branch ${COMFYUI_VERSION} https://github.com/comfyanonymous/ComfyUI.git /workspace

RUN mkdir -p /cache/tunableop /cache/triton /cache/miopen

RUN git clone --depth 1 https://github.com/Comfy-Org/ComfyUI-Manager /workspace/custom_nodes/ComfyUI-Manager || true
RUN sleep 2 && git clone --depth 1 https://github.com/patientx/ComfyUI-INT8-Fast-ROCM /workspace/custom_nodes/ComfyUI-INT8-Fast-ROCM || true
RUN sleep 2 && git clone --depth 1 https://github.com/iGavroche/rocm-ninodes /workspace/custom_nodes/rocm-ninodes || true
RUN sleep 2 && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF /workspace/custom_nodes/ComfyUI-GGUF || true
RUN sleep 2 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes /workspace/custom_nodes/ComfyUI-KJNodes || true
RUN sleep 2 && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /workspace/custom_nodes/ComfyUI-VideoHelperSuite || true
RUN sleep 2 && git clone --depth 1 https://github.com/rgthree/rgthree-comfy /workspace/custom_nodes/rgthree-comfy || true
RUN sleep 2 && git clone --depth 1 https://github.com/chflame163/ComfyUI_LayerStyle /workspace/custom_nodes/ComfyUI_LayerStyle || true
RUN sleep 2 && git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler /workspace/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler || true

# ponytail: pin ROCm torch to prevent CUDA overwrites during runtime updates
RUN echo "torch==2.12.0+rocm7.14" > /opt/venv/pip-constraints.txt

# 1단계: 일반 PyPI 패키지 및 ComfyUI 관련 기본 요구사항 설치 (기본 PyPI 서버 사용)
RUN pip install --no-cache-dir -r /workspace/requirements.txt && \
    find /workspace/custom_nodes -name "requirements.txt" -exec pip install --no-cache-dir -r {} \; 2>/dev/null || true && \
    pip install --no-cache-dir gguf comfyui-manager==4.2.2 && \
    pip install --no-cache-dir comfy-kitchen==0.2.28

# ponytail: comfy-kitchen 0.2.30 HIP 백엔드(gfx1100)에서 int8_convrot 연산이
# 단색 영상 출력을 유발 (실측 검증: 0.2.28 정상 / 0.2.30 단색). 0.2.28 고정.
# v0.32.0 코어 attention.py는 0.2.30에 추가된 int8_attention_is_available()을
# import 시점에 호출하므로, 0.2.28에는 없는 API를 가드.
RUN python -c "src=open('/workspace/comfy/ldm/modules/attention.py').read(); old='COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = comfy_kitchen.int8_attention_is_available()'; new='try:\n    COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = comfy_kitchen.int8_attention_is_available()\nexcept AttributeError:\n    COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = False'; assert old in src, 'attention.py pattern not found'; open('/workspace/comfy/ldm/modules/attention.py','w').write(src.replace(old,new)); print('attention.py comfy-kitchen guard patched')"

# ponytail: Triton-only flash-attn install before constraint activation.
# FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE skips HIP C++ extension
# (flash_attn_2_cuda, CDNA-only) and builds only the Triton backend.
RUN FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    CUDA_HOME=/opt/rocm \
    pip install --no-cache-dir flash-attn==2.8.3.post1 --no-build-isolation

# ponytail: SageAttention v2.2.0 + thu-ml PR #381 (ROCm/HIP support) applied on top.
# Official repo clone (main d1a57a54 == PR base) + `curl .../pull/381.patch | git apply`.
# PR #381: setup.py HIP auto-detect (skip CUDA ext), sageattn() routes to Triton
# kernels, num_stages=1 under HIP (avoids AMD Triton pipelining UAF #365; 2.8x
# faster than num_stages=4 on RDNA3.5). Measured on ROCm 7.14 + torch 2.12.0 +
# triton 3.7.1 (same as this image): 1.16-1.33x over FA2 Triton at seq 8k-32k.
RUN pip install --no-cache-dir packaging && \
    git clone https://github.com/thu-ml/SageAttention.git /tmp/sageattn && \
    cd /tmp/sageattn && \
    git checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 && \
    curl -sL https://github.com/thu-ml/SageAttention/pull/381.diff | git apply - && \
    git diff --stat && \
    pip install --no-build-isolation --no-cache-dir . && \
    cd / && rm -rf /tmp/sageattn

# ponytail: aiter v0.1.13 from source — ROCm-optimized attention kernels matching rocm-ninodes
RUN git clone --depth=1 --branch v0.1.13 --recursive \
    https://github.com/ROCm/aiter.git /opt/aiter \
    && cd /opt/aiter \
    && GPU_ARCHS=gfx1100 MAX_JOBS=8 AITER_USE_SYSTEM_TRITON=1 \
       pip install --no-build-isolation . \
    && rm -rf /opt/aiter

# ponytail: prebuild aiter_core JIT module into the image (runtime JIT build
# races on lock_module_aiter_core — wedges container boot when a build dies)
RUN GPU_ARCHS=gfx1100 python -c "import aiter"

# now activate constraints for any future runtime installs
ENV PIP_CONSTRAINT=/opt/venv/pip-constraints.txt

# ponytail: ComfyUI-Manager ROCm protection - prevent CUDA torch overwrite
RUN mkdir -p /workspace/user/__manager && \
    printf 'torch\ntriton\nflash-attn\ntorchaudio\ntorchvision\naiter\ncomfy-kitchen\nsageattention\n' > /workspace/user/__manager/pip_blacklist.list && \
    printf '[default]\nmodel_download_by_agent = True\nsecurity_level = weak\nnetwork_mode = personal_cloud\nallow_git_url_install = True\nallow_pip_install = True\n' > /workspace/user/__manager/config.ini

# 커스텀 노드 중 의존성 필요한 것들 설치
RUN pip install --no-cache-dir timm ultralytics opencv-contrib-python pymatting --no-deps
RUN pip install --no-cache-dir segment-anything piexif webcolors insightface llama_cpp-python

# ponytail: SeedVR2 deps — --no-deps to bypass pip resolver (ResolutionImpossible on ROCm torch); numpy/einops/etc already in image
RUN pip install --no-cache-dir omegaconf diffusers peft accelerate rotary_embedding_torch --no-deps

RUN mkdir -p /workspace/user/__manager && \
    printf 'torch\ntriton\nflash-attn\ntorchaudio\ntorchvision\ntimm\naiter\ncomfy-kitchen\nsageattention\n' > /workspace/user/__manager/pip_blacklist.list

RUN echo '#!/bin/bash\nset -e\nmkdir -p /workspace/user/__manager\nprintf "[default]\\nmodel_download_by_agent = True\\nsecurity_level = weak\\nnetwork_mode = personal_cloud\\nallow_git_url_install = True\\nallow_pip_install = True\\n" > /workspace/user/__manager/config.ini\nprintf "torch\\ntriton\\nflash-attn\\naiter\\ncomfy-kitchen\\nsageattention\\n" > /workspace/user/__manager/pip_blacklist.list\nexec python main.py \\\n    --listen 0.0.0.0 \\\n    --port ${PORT:-8188} \\\n    --disable-api-nodes \\\n    --enable-manager \\\n    --enable-manager-legacy-ui \\\n    ${CLI_ARGS}' > /opt/entrypoint.sh

RUN chmod +x /opt/entrypoint.sh

VOLUME ["/workspace/models", "/workspace/output", "/workspace/user"]
EXPOSE 8188

ENTRYPOINT ["/opt/entrypoint.sh"]
