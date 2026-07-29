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
    AITER_TRITON_ONLY=1 \
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8 \
    MALLOC_MMAP_THRESHOLD_=65536 \
    MALLOC_TRIM_THRESHOLD_=65536

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
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

ARG COMFYUI_VERSION=v0.28.2
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
# ponytail: pure Triton FA-2 for RDNA3 — gfx1100 optimized forward/backward/decode
RUN sleep 2 && git clone --depth 1 https://github.com/chelokot/flash-attention-rdna3.git /workspace/custom_nodes/RDNA3-Flash-Attention || true

# ponytail: pin ROCm torch to prevent CUDA overwrites during runtime updates
RUN echo "torch==2.12.0+rocm7.14" > /opt/venv/pip-constraints.txt

# 1단계: 일반 PyPI 패키지 및 ComfyUI 관련 기본 요구사항 설치 (기본 PyPI 서버 사용)
RUN pip install --no-cache-dir -r /workspace/requirements.txt && \
    find /workspace/custom_nodes -name "requirements.txt" -exec pip install --no-cache-dir -r {} \; 2>/dev/null || true && \
    pip install --no-cache-dir gguf comfyui-manager==4.2.2 && \
    pip install --no-cache-dir --upgrade comfy-kitchen==0.2.22

# ponytail: Triton-only flash-attn install before constraint activation.
# FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE skips HIP C++ extension
# (flash_attn_2_cuda, CDNA-only) and builds only the Triton backend.
RUN FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    CUDA_HOME=/opt/rocm \
    pip install --no-cache-dir flash-attn==2.8.3.post1 --no-build-isolation

# ponytail: fa-rdna3 — pure Triton FA-2 for RDNA3 (editable install so ComfyUI node resolves imports)
RUN pip install --no-cache-dir -e /workspace/custom_nodes/RDNA3-Flash-Attention

# ponytail: aiter v0.1.13 from source — ROCm-optimized attention kernels matching rocm-ninodes
RUN git clone --depth=1 --branch v0.1.13 --recursive \
    https://github.com/ROCm/aiter.git /opt/aiter \
    && cd /opt/aiter \
    && GPU_ARCHS=gfx1100 MAX_JOBS=8 AITER_USE_SYSTEM_TRITON=1 \
       pip install --no-build-isolation . \
    && rm -rf /opt/aiter

# now activate constraints for any future runtime installs
ENV PIP_CONSTRAINT=/opt/venv/pip-constraints.txt

# ponytail: ComfyUI-Manager ROCm protection - prevent CUDA torch overwrite
RUN mkdir -p /workspace/user/__manager && \
    printf 'torch\ntriton\nflash-attn\ntorchaudio\ntorchvision\naiter\n' > /workspace/user/__manager/pip_blacklist.list && \
    printf '[default]\nmodel_download_by_agent = True\nsecurity_level = weak\nnetwork_mode = personal_cloud\nallow_git_url_install = True\nallow_pip_install = True\n' > /workspace/user/__manager/config.ini

# 커스텀 노드 중 의존성 필요한 것들 설치
RUN pip install --no-cache-dir timm ultralytics opencv-contrib-python pymatting --no-deps
RUN pip install --no-cache-dir segment-anything piexif webcolors insightface llama_cpp-python

RUN mkdir -p /workspace/user/__manager && \
    printf 'torch\ntriton\nflash-attn\ntorchaudio\ntorchvision\ntimm\naiter\n' > /workspace/user/__manager/pip_blacklist.list

RUN echo '#!/bin/bash\nset -e\nmkdir -p /workspace/user/__manager\nprintf "[default]\\nmodel_download_by_agent = True\\nsecurity_level = weak\\nnetwork_mode = personal_cloud\\nallow_git_url_install = True\\nallow_pip_install = True\\n" > /workspace/user/__manager/config.ini\nprintf "torch\\ntriton\\nflash-attn\\naiter\\n" > /workspace/user/__manager/pip_blacklist.list\nexec python main.py \\\n    --listen 0.0.0.0 \\\n    --port ${PORT:-8188} \\\n    --disable-api-nodes \\\n    --cache-none \\\n    --disable-mmap \\\n    --enable-manager \\\n    --enable-manager-legacy-ui \\\n    ${CLI_ARGS}' > /opt/entrypoint.sh

RUN chmod +x /opt/entrypoint.sh

VOLUME ["/workspace/models", "/workspace/output", "/workspace/user"]
EXPOSE 8188

ENTRYPOINT ["/opt/entrypoint.sh"]
