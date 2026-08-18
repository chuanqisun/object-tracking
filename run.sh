#!/usr/bin/env bash
set -e

# Resolve repository and inference directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/inference/server.cpp" ]; then
    ROOT_DIR="$SCRIPT_DIR"
    INFERENCE_DIR="$SCRIPT_DIR/inference"
elif [ -f "$SCRIPT_DIR/server.cpp" ]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    INFERENCE_DIR="$SCRIPT_DIR"
else
    ROOT_DIR="$PWD"
    INFERENCE_DIR="$PWD/inference"
fi

EXECUTABLE="$INFERENCE_DIR/yolo_server"

# Auto-build if yolo_server binary does not exist
if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable $EXECUTABLE not found. Running build script first..."
    if [ -f "$ROOT_DIR/build.sh" ]; then
        "$ROOT_DIR/build.sh"
    elif [ -f "$INFERENCE_DIR/build.sh" ]; then
        "$INFERENCE_DIR/build.sh"
    else
        echo "Error: Build script not found!"
        exit 1
    fi
fi

PARAM_PATH="${1:-$ROOT_DIR/puck-eye-seg-s_ncnn_model/model.ncnn.param}"
BIN_PATH="${2:-$ROOT_DIR/puck-eye-seg-s_ncnn_model/model.ncnn.bin}"
PORT="${3:-18888}"

if [ ! -f "$PARAM_PATH" ]; then
    echo "Error: Param file not found at: $PARAM_PATH"
    exit 1
fi

if [ ! -f "$BIN_PATH" ]; then
    echo "Error: Bin file not found at: $BIN_PATH"
    exit 1
fi

echo "=========================================="
echo " Starting NCNN Vulkan C++ Inference Server"
echo " Param: $PARAM_PATH"
echo " Bin:   $BIN_PATH"
echo " Port:  $PORT"
echo "=========================================="

exec "$EXECUTABLE" "$PARAM_PATH" "$BIN_PATH" "$PORT"
