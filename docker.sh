#!/usr/bin/env bash

sudo docker run \
	--rm \
	-it \
	--network=host \
	--device=/dev/kfd \
	--device=/dev/dri \
	--group-add=video \
	--ipc=host \
	--cap-add=SYS_PTRACE \
	--security-opt seccomp=unconfined \
	-v "$(pwd):/ai-pytorch" \
	rocm/pytorch

# source ai-venv/bin/activate
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0
