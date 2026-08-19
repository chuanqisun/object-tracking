
#!/usr/bin/env bash
set -e

echo "=== 1. Installing System Dependencies ==="

if command -v dnf &>/dev/null; then
    echo "Detected dnf (Fedora/RHEL). Installing packages..."
    sudo dnf install -y \
        gcc-c++ make git pkg-config \
        onnxruntime-devel turbojpeg-devel opencv-devel zlib-devel openssl-devel
elif command -v apt-get &>/dev/null; then
    echo "Detected apt-get (Ubuntu/Debian). Installing packages..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential git pkg-config \
        libonnxruntime-dev libturbojpeg0-dev libopencv-dev zlib1g-dev libssl-dev
elif command -v pacman &>/dev/null; then
    echo "Detected pacman (Arch Linux). Installing packages..."
    sudo pacman -S --needed --noconfirm \
        base-devel git pkgconf \
        onnxruntime libjpeg-turbo opencv zlib openssl
else
    echo "[WARNING] Package manager not recognized. Please ensure gcc-c++, make, git, pkg-config,"
    echo "          onnxruntime-devel, turbojpeg-devel, opencv-devel, and zlib-devel are installed."
fi

echo "=== 2. Building and Installing uSockets ==="
if [ ! -d "uSockets" ]; then
    git clone --recursive https://github.com/uNetworking/uSockets.git
fi
cd uSockets
make
sudo cp src/libusockets.h /usr/include/

if [ -d "/usr/lib64" ]; then
    sudo cp uSockets.a /usr/lib64/libusockets.a
else
    sudo cp uSockets.a /usr/lib/libusockets.a
fi
cd ..

echo "=== 3. Installing uWebSockets ==="
if [ ! -d "uWebSockets" ]; then
    git clone --recursive https://github.com/uNetworking/uWebSockets.git
fi
sudo mkdir -p /usr/include/uWebSockets
sudo cp -r uWebSockets/src/* /usr/include/uWebSockets/
sudo cp -r uWebSockets/src/* /usr/include/

echo "=== [SUCCESS] All dependencies for server_npu.cpp installed successfully! ==="