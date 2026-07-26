IMAGE_NAME ?= comfyui-rocm
VERSION = $(shell grep '^FROM' Dockerfile | awk '{print $$2}' | sed 's|rocm/pytorch:rocm||;s|_py.*||' | sed 's|_ubuntu.*||')
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
	docker run --rm --entrypoint sh $(FULL_TAG) -c "python -c 'import torch; print(f\"torch: {torch.__version__}\")' && python -c 'import triton; print(f\"triton: {triton.__version__}\")' && python -c 'import flash_attn; print(f\"flash_attn: {flash_attn.__version__}\")'"

restart:
	docker compose down && docker compose up -d
