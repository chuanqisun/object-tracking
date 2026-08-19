#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ORT_DIR="/home/stack/repos/onnxruntime-build/Release"
ORT_INC="/home/stack/repos/onnxruntime-src/include/onnxruntime/core/session"

g++ -std=c++20 -O3 -march=native -flto \
    server_npu.cpp -o puck_eye_npu_server \
    -I/usr/include/uWebSockets \
    -I"${ORT_INC}" \
    -L"${ORT_DIR}" \
    -Wl,-rpath,"${ORT_DIR}" \
    -Wl,-rpath,'$ORIGIN/lib' \
    $(pkg-config --cflags --libs libturbojpeg) \
    -lusockets -lonnxruntime -lz -lpthread

echo "[SUCCESS] puck_eye_npu_server built successfully."