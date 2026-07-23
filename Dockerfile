FROM rocm/pytorch:rocm7.14_ubuntu26.04_py3.14_pytorch_release_2.12.0

ENV MIOPEN_FIND_ENFORCE=1 \
    MIOPEN_FIND_MODE=FAST \
    MIOPEN_DEBUG_DISABLE_FIND_DB=0 \
    PYTORCH_TUNABLEOP_CACHE_DIR=/cache/tunableop \
    TRITON_CACHE_DIR=/cache/triton \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    COMFYUI_ENABLE_MIOpen=1 \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    AITER_TRITON_ONLY=1 \
    MALLOC_MMAP_THRESHOLD_=65536 \
    MALLOC_TRIM_THRESHOLD_=65536

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libx11-6 \
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

RUN pip install --no-cache-dir -r /workspace/requirements.txt && \
    find /workspace/custom_nodes -name "requirements.txt" -exec pip install --no-cache-dir -r {} \; && \
    pip install --no-cache-dir gguf comfyui-manager==4.2.2 && \
    pip install --no-cache-dir --upgrade comfy-kitchen==0.2.22

# ponytail: save ROCm triton binary before flash-attn install.
# flash-attn's setup.py (via aiter subprocess) replaces the ROCm-optimized
# triton (884MB libtriton.so) with a generic pip version (461MB) that
# segfaults when imported after torch. Restore the ROCm binary after install.
RUN cp /opt/venv/lib/python3.14/site-packages/triton/_C/libtriton.so /tmp/libtriton_rocm.so && \
    git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
    cd /tmp/flash-attention && \
    sed -i '/subprocess.run.*pip.*install.*third_party\/aiter/s/^/#/' setup.py && \
    pip install --no-cache-dir packaging ninja einops && \
    pip install --no-cache-dir --no-build-isolation --no-deps . && \
    cp /tmp/libtriton_rocm.so /opt/venv/lib/python3.14/site-packages/triton/_C/libtriton.so && \
    cd / && rm -rf /tmp/flash-attention /tmp/libtriton_rocm.so

RUN echo '#!/bin/bash\nset -e\nexec python main.py \\\n    --listen 0.0.0.0 \\\n    --port ${PORT:-8188} \\\n    --disable-api-nodes \\\n    --cache-none \\\n    --disable-smart-memory \\\n    --disable-pinned-memory \\\n    --enable-manager \\\n    --enable-manager-legacy-ui \\\n    ${COMFYUI_ARGS}' > /opt/entrypoint.sh

RUN chmod +x /opt/entrypoint.sh

VOLUME ["/workspace/models", "/workspace/output", "/workspace/user"]
EXPOSE 8188

ENTRYPOINT ["/opt/entrypoint.sh"]
