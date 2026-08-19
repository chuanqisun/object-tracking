#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT="${1:-18888}"
IMAGE_NAME="ryzenai-runtime:24.04"

echo "[INFO] Starting Ryzen AI NPU Server Container on port ${PORT}..."
docker run --rm -d \
    --name ryzenai_npu_server \
    --privileged \
    --ipc=host \
    --net=host \
    -v /dev:/dev \
    -v /sys:/sys \
    -v "$PWD/..":/workspace \
    -w /workspace/npu \
    "$IMAGE_NAME" \
    bash -c "
        source /opt/xilinx/xrt/setup.sh
        /opt/xilinx/xrt/bin/xrt-smi examine
        python3 server.py
    "

echo "[SUCCESS] Container started in background (ryzenai_npu_server)."
