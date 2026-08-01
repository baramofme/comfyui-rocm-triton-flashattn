IMAGE_NAME ?= comfyui-rocm
BASE_ROCM = $(shell grep '^FROM' Dockerfile | awk '{print $$2}' | sed 's|rocm/pytorch:rocm||;s|_py.*||' | sed 's|_ubuntu.*||')
VERSION = rocm$(BASE_ROCM)-py3.12-torch2.12.0-triton3.7.1-fa2.8.3-aiter0.1.13-comfy0.28.2
FULL_TAG = $(IMAGE_NAME):$(VERSION)
LATEST_TAG = $(IMAGE_NAME):latest

.PHONY: build tag push clean logs shell test

build:
	DOCKER_BUILDKIT=0 docker build -t $(FULL_TAG) .

tag: build
	docker tag $(FULL_TAG) $(LATEST_TAG)

push:
	docker push $(FULL_TAG)
	docker push $(LATEST_TAG)

clean:
	docker rmi $(FULL_TAG) $(LATEST_TAG) 2>/dev/null || true

logs:
	docker logs -f comfyui_gpu0

shell:
	docker exec -it comfyui_gpu0 bash

test:
	docker run --rm \
		--device /dev/kfd --device /dev/dri \
		--group-add video \
		--entrypoint sh $(FULL_TAG) -c "\
		python -c 'import torch; print(f\"torch: {torch.__version__}\")' && \
		python -c 'import triton; print(f\"triton: {triton.__version__}\")' && \
		readlink -f /opt/venv/lib/python3.12/site-packages/torch/lib/libaotriton_v2.so | grep -oP 'libaotriton_v2\.so\.\K.*' | xargs -I{} echo \"AOTriton: {}\" && \
		python -c 'import flash_attn; print(f\"flash_attn: {flash_attn.__version__}\")' && \
		python -c 'import aiter; print(\"aiter: ok\")' \
	"

restart:
	docker compose down && docker compose up -d
