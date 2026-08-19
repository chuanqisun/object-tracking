#!/usr/bin/env bash
set -e

# Resolve repository and inference directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/server.cpp" ]; then
    INFERENCE_DIR="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/inference" ]; then
    INFERENCE_DIR="$SCRIPT_DIR/inference"
else
    echo "Error: Could not locate inference directory with server.cpp."
    exit 1
fi

echo "=========================================================="
echo " Building Optimized NCNN Vulkan Server (Zen 5 / RDNA 3.5) "
echo "=========================================================="

# 1. Setup PKG_CONFIG_PATH for NCNN
NCNN_PATHS=(
    "/home/stack/repos/ncnn/build/install/lib64/pkgconfig"
    "/home/stack/repos/ncnn/build/install/lib/pkgconfig"
    "$SCRIPT_DIR/../ncnn/build/install/lib64/pkgconfig"
    "$SCRIPT_DIR/../ncnn/build/install/lib/pkgconfig"
    "$SCRIPT_DIR/ncnn/build/install/lib64/pkgconfig"
    "$SCRIPT_DIR/ncnn/build/install/lib/pkgconfig"
)

for p in "${NCNN_PATHS[@]}"; do
    if [ -d "$p" ]; then
        export PKG_CONFIG_PATH="$p:$PKG_CONFIG_PATH"
    fi
done

# 2. Check for crow_all.h header
if [ ! -f "$INFERENCE_DIR/crow_all.h" ]; then
    echo "--> crow_all.h not found. Downloading Crow v1.2.0 single header..."
    curl -L -o "$INFERENCE_DIR/crow_all.h" https://github.com/CrowCpp/Crow/releases/download/v1.2.0/crow_all.h
fi

# 3. Check for dependencies
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ compiler not found."
    exit 1
fi

if ! pkg-config --exists opencv4; then
    echo "Error: opencv4 pkg-config not found."
    exit 1
fi

if ! pkg-config --exists ncnn; then
    echo "Error: ncnn pkg-config not found. Make sure NCNN is installed or PKG_CONFIG_PATH is set correctly."
    exit 1
fi

# Check for libturbojpeg optional acceleration
EXTRA_LIBS=""
if pkg-config --exists libturbojpeg; then
    echo "--> Found libturbojpeg via pkg-config. TurboJPEG SIMD decoding enabled."
    EXTRA_LIBS="$(pkg-config --cflags --libs libturbojpeg)"
elif [ -f "/usr/include/turbojpeg.h" ] || [ -f "/usr/local/include/turbojpeg.h" ]; then
    echo "--> Found libturbojpeg header. Linking -lturbojpeg."
    EXTRA_LIBS="-lturbojpeg"
else
    echo "--> Notice: libturbojpeg header not found (using OpenCV imdecode fallback). Install turbojpeg-devel (Fedora) or libturbojpeg0-dev (Ubuntu/Debian) for maximum speed."
fi

# 4. Compile server.cpp
echo "--> Compiling server.cpp into yolo_server with Zen 5 + AVX-512 optimizations..."
cd "$INFERENCE_DIR"

g++ -O3 -march=native -flto -fopenmp \
    server.cpp -o yolo_server \
    $(pkg-config --cflags --libs opencv4 ncnn) \
    $EXTRA_LIBS \
    -L/home/stack/repos/ncnn/build/install/lib64 \
    -lglslang -lMachineIndependent -lGenericCodeGen -lOSDependent -lSPIRV \
    -lvulkan -lpthread

echo "----------------------------------------------------------"
echo " Build successful!"
echo " Executable created at: $INFERENCE_DIR/yolo_server"
echo "----------------------------------------------------------"
