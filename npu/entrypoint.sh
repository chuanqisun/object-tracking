#!/bin/bash
set -e

# Source XRT setup if present
if [ -f /opt/xilinx/xrt/setup.sh ]; then
    source /opt/xilinx/xrt/setup.sh
fi

# Source Ryzen AI virtualenv if present
if [ -f /opt/ryzen-ai/venv/bin/activate ]; then
    source /opt/ryzen-ai/venv/bin/activate
fi

# Preload libxrt_coreutil.so.2 to resolve XRT core symbol dependencies
export LD_PRELOAD="/opt/xilinx/xrt/lib/libxrt_coreutil.so.2:${LD_PRELOAD}"

DEFAULT_MODEL="/workspace/yolo26s-seg_qdq.onnx"
if [ ! -f "$DEFAULT_MODEL" ]; then
    DEFAULT_MODEL="/workspace/yolo26s-seg.onnx"
fi

DEFAULT_VAIP_CONFIG="/opt/ryzen-ai/venv/lib/python3.12/site-packages/voe/bin/vaip_config.json"
if [ ! -f "$DEFAULT_VAIP_CONFIG" ]; then
    DEFAULT_VAIP_CONFIG="/workspace/vaiml_config.json"
fi

export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
export VAIML_CONFIG="${VAIML_CONFIG:-$DEFAULT_VAIP_CONFIG}"
export NPU_CACHE_DIR="${NPU_CACHE_DIR:-/workspace/npu_cache}"
export PORT="${PORT:-8765}"

mkdir -p "$NPU_CACHE_DIR" /workspace/profiles

echo "=================================================="
echo " Starting Turn-Key YOLO26 NPU Inference Server"
echo " Model Path:   $MODEL_PATH"
echo " VAIML Config: $VAIML_CONFIG"
echo " Listening:    ws://0.0.0.0:$PORT"
echo "=================================================="

if [ "$#" -gt 0 ]; then
    exec "$@"
else
    exec python /workspace/server.py
fi
