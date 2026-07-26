FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

ENV MIOPEN_FIND_ENFORCE=1 \
    MIOPEN_FIND_MODE=FAST \
    MIOPEN_DEBUG_DISABLE_FIND_DB=0 \
    PYTORCH_TUNABLEOP_CACHE_DIR=/cache/tunableop \
    TRITON_CACHE_DIR=/cache/triton \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    COMFYUI_ENABLE_MIOpen=1 \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    AITER_TRITON_ONLY=1 \
    AITER_USE_SYSTEM_TRITON=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    MALLOC_MMAP_THRESHOLD_=65536 \
    MALLOC_TRIM_THRESHOLD_=65536

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
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

# ponytail: pin ROCm torch/triton to prevent CUDA overwrites during runtime updates
RUN echo "torch==2.10.0+rocm7.2.4" > /opt/venv/pip-constraints.txt && \
    echo "triton==3.3.1" >> /opt/venv/pip-constraints.txt

# install requirements WITHOUT constraints (constraints pin to +rocm version not on PyPI, pip can't resolve)
RUN pip install --no-cache-dir -r /workspace/requirements.txt && \
    find /workspace/custom_nodes -name "requirements.txt" -exec pip install --no-cache-dir -r {} \; 2>/dev/null || true && \
    pip install --no-cache-dir gguf comfyui-manager==4.2.2 && \
    pip install --no-cache-dir --upgrade comfy-kitchen==0.2.22

# now activate constraints for any future runtime installs
ENV PIP_CONSTRAINT=/opt/venv/pip-constraints.txt

# ponytail: ComfyUI-Manager ROCm protection - prevent CUDA torch overwrite
RUN mkdir -p /workspace/user/__manager && \
    printf 'torch\ntriton\nflash-attn\n' > /workspace/user/__manager/pip_blacklist.list && \
    printf '[default]\nmodel_download_by_agent = True\nsecurity_level = weak\nnetwork_mode = personal_cloud\nallow_git_url_install = True\nallow_pip_install = True\n' > /workspace/user/__manager/config.ini

# ponytail: save ROCm triton binary before flash-attn install.
# flash-attn's setup.py (via aiter subprocess) replaces the ROCm-optimized
# triton (884MB libtriton.so) with a generic pip version (461MB) that
# segfaults when imported after torch. Restore the ROCm binary after install.
RUN cp /opt/venv/lib/python3.12/site-packages/triton/_C/libtriton.so /tmp/libtriton_rocm.so && \
    git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
    cd /tmp/flash-attention && \
    sed -i '/subprocess.run.*pip.*install.*third_party\/aiter/s/^/#/' setup.py && \
    pip install --no-cache-dir packaging ninja einops pybind11 && \
    pip install --no-cache-dir --no-build-isolation --no-deps . && \
    cp /tmp/libtriton_rocm.so /opt/venv/lib/python3.12/site-packages/triton/_C/libtriton.so && \
    cd / && rm -rf /tmp/flash-attention /tmp/libtriton_rocm.so

RUN echo '#!/bin/bash\nset -e\nmkdir -p /workspace/user/__manager\nprintf "[default]\\nmodel_download_by_agent = True\\nsecurity_level = weak\\nnetwork_mode = personal_cloud\\nallow_git_url_install = True\\nallow_pip_install = True\\n" > /workspace/user/__manager/config.ini\nprintf "torch\\ntriton\\nflash-attn\\n" > /workspace/user/__manager/pip_blacklist.list\nexec python main.py \\\n    --listen 0.0.0.0 \\\n    --port ${PORT:-8188} \\\n    --disable-api-nodes \\\n    --cache-none \\\n    --disable-mmap \\\n    --enable-manager \\\n    --enable-manager-legacy-ui \\\n    ${CLI_ARGS}' > /opt/entrypoint.sh

RUN chmod +x /opt/entrypoint.sh

VOLUME ["/workspace/models", "/workspace/output", "/workspace/user"]
EXPOSE 8188

ENTRYPOINT ["/opt/entrypoint.sh"]
