#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL_PATH="${1:-puck-eye-seg-s_bf16_ctx.onnx}"
PORT="${2:-18888}"

# Load XRT environment
if [ -f /opt/xilinx/xrt/setup.sh ]; then
    source /opt/xilinx/xrt/setup.sh
elif [ -f /usr/xrt/setup.sh ]; then
    source /usr/xrt/setup.sh
else
    echo "[FATAL] XRT setup.sh was not found."
    exit 1
fi

# Determine ONNX Runtime library directory
if [ -z "${AMD_ORT_LIB:-}" ]; then
    if [ -n "${RYZEN_AI_INSTALLATION_PATH:-}" ] && [ -d "${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/lib" ]; then
        AMD_ORT_LIB="${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/lib"
    elif [ -d "/home/stack/repos/onnxruntime-build/Release" ]; then
        AMD_ORT_LIB="/home/stack/repos/onnxruntime-build/Release"
    else
        echo "[FATAL] Cannot locate ONNX Runtime library directory."
        exit 1
    fi
fi

ORT_SO="${AMD_ORT_LIB}/libonnxruntime.so"
test -e "$ORT_SO" || {
    echo "[FATAL] Missing $ORT_SO"
    exit 1
}

# Verify Vitis AI Provider library
VITIS_PROVIDER="${AMD_ORT_LIB}/libonnxruntime_providers_vitisai.so"
if [ ! -f "$VITIS_PROVIDER" ]; then
    echo "[FATAL] libonnxruntime_providers_vitisai.so was not found in $AMD_ORT_LIB"
    exit 1
fi

export LD_LIBRARY_PATH="${AMD_ORT_LIB}:/opt/xilinx/xrt/lib:/opt/xilinx/xrt/lib64:${LD_LIBRARY_PATH:-}"

echo "[INFO] ONNX Runtime library: $AMD_ORT_LIB"
echo "[INFO] Executable dependencies:"
ldd ./puck_eye_npu_server | grep -E 'onnxruntime|vitis|ryzen|xrt' || true

# Confirm XRT sees NPU
if command -v xrt-smi >/dev/null 2>&1; then
    xrt-smi examine || true
elif [ -x /opt/xilinx/xrt/bin/xrt-smi ]; then
    /opt/xilinx/xrt/bin/xrt-smi examine || true
else
    echo "[FATAL] xrt-smi is unavailable"
    exit 1
fi

# Model check
if [ ! -f "$MODEL_PATH" ]; then
    if [ -f puck-eye-seg-s.onnx ]; then
        echo "[WARNING] $MODEL_PATH not found; using puck-eye-seg-s.onnx"
        MODEL_PATH="puck-eye-seg-s.onnx"
    else
        echo "[FATAL] Model not found: $MODEL_PATH"
        exit 1
    fi
fi

test -f vaip_bf16_config.json || {
    echo "[FATAL] Missing vaip_bf16_config.json"
    exit 1
}

rm -rf ./npu_cache_bf16

exec ./puck_eye_npu_server "$MODEL_PATH" "$PORT"