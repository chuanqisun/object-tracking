# NCNN C++ GPU Inference Server (Radeon 890M / RDNA 3.5 Vulkan)

High-performance, low-latency C++ inference server for YOLO26 segmentation models using **NCNN + Vulkan Compute** on Linux (Fedora / RDNA 3.5 iGPU).

---

## Overview

This C++ backend inference server leverages **NCNN** with Vulkan acceleration and **Crow** (a lightweight micro web framework) to provide high-throughput, low-latency object detection and instance segmentation inference over WebSocket and HTTP POST endpoints.

- **Engine:** NCNN (Vulkan Compute API, FP16/Packed)
- **Server:** Crow (Header-only micro-framework built on Asio)
- **Image Processing:** OpenCV 4
- **Endpoints:** WebSocket (`/ws`) for streaming inference and HTTP POST (`/predict`) with CORS support.
- **Target Hardware:** AMD Radeon 890M iGPU (RDNA 3.5 / GFX1150) or any Vulkan-capable GPU / CPU.

---

## 1. Prerequisites & System Dependencies

### Fedora Linux

Install build tools, Vulkan development headers, OpenCV, Asio, and glslang:

```bash
sudo dnf install -y \
    gcc-c++ cmake git \
    vulkan-loader-devel mesa-vulkan-drivers vulkan-tools \
    glslang glslc opencv opencv-devel zlib-devel openssl-devel \
    asio-devel boost-devel
```

### Verify Vulkan Device

Ensure your RADV Vulkan driver detects the Radeon 890M iGPU:

```bash
vulkaninfo --summary
# Look for: GPU0: ... AMD Radeon Graphics (RADV STRIX1)
```

---

## 2. Build NCNN with Vulkan Acceleration

Clone and build NCNN with Vulkan acceleration enabled:

```bash
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=ON \
      -DNCNN_SYSTEM_GLSLANG=ON \
      -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_TESTS=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF ..

make -j$(nproc)
make install
cd ../..
```

This builds NCNN static libraries and headers into `ncnn/build/install/`.

---

## 3. Setup Crow Web Framework

Download the single-header Crow release inside the `inference` folder:

```bash
cd inference
curl -L -o crow_all.h https://github.com/CrowCpp/Crow/releases/download/v1.2.0/crow_all.h
```

---

## 4. Build the C++ Inference Server

You can build using the `build.sh` script:

```bash
./build.sh
```

Or manually set `PKG_CONFIG_PATH` to point to the NCNN pkgconfig directory and compile `server.cpp`:

```bash
export PKG_CONFIG_PATH=/home/stack/repos/ncnn/build/install/lib64/pkgconfig:$PKG_CONFIG_PATH

cd inference

g++ -O3 -march=native -fopenmp \
    server.cpp -o yolo_server \
    `pkg-config --cflags --libs opencv4 ncnn` \
    -L/home/stack/repos/ncnn/build/install/lib64 \
    -lglslang -lMachineIndependent -lGenericCodeGen -lOSDependent -lSPIRV \
    -lvulkan -lpthread
```

---

## 5. Running the Inference Server

You can start the server using `run.sh`:

```bash
./run.sh [param_path] [bin_path] [port]
```

Or run `yolo_server` directly by passing the NCNN `.param` file path, `.bin` file path, and an optional port number (default: `18888`):

```bash
# Usage: ./yolo_server <path_to_param> <path_to_bin> [port]

cd inference
./yolo_server ../puck-eye-seg-s_ncnn_model/model.ncnn.param ../puck-eye-seg-s_ncnn_model/model.ncnn.bin 18888
```

When started, the server outputs:

```text
Loading NCNN model: ../puck-eye-seg-s_ncnn_model/model.ncnn.param and ../puck-eye-seg-s_ncnn_model/model.ncnn.bin
Server starting on http://localhost:18888 (WebSocket at ws://localhost:18888/ws)
```

---

## 6. Server Endpoints & API Protocol

### WebSocket Stream Endpoint (`ws://localhost:18888/ws`)

- **Input:** Raw image binary buffer (JPEG / PNG byte array).
- **Output:** JSON string with detection bounding boxes and GPU execution latency (`infer_ms`).

### HTTP POST Endpoint (`http://localhost:18888/predict`)

- **Method:** `POST` (supports `OPTIONS` for CORS pre-flight)
- **Body:** Binary image data (`Content-Type: application/octet-stream` or `image/jpeg`)
- **Output:** JSON response containing detections, image dimensions, and GPU inference time.

---

## 7. Testing the Server

### Testing HTTP Endpoint with `curl`

Send an image via HTTP POST to verify model execution:

```bash
curl -s -X POST --data-binary @sample.jpg http://localhost:18888/predict
```

#### Example Response JSON

```json
{
  "img_w": 640,
  "img_h": 640,
  "target_size": 640,
  "infer_ms": 3.42,
  "detections": [
    {
      "class_id": 0,
      "score": 0.745117,
      "box": [247.125, 318.5, 417.5, 396.5]
    }
  ]
}
```

---

## File Structure

```
inference/
├── server.cpp       # C++ NCNN Vulkan WebSocket & HTTP inference server
├── crow_all.h       # Crow micro web framework header
├── yolo_server      # Compiled binary executable
└── README.md        # Server setup & execution guide
```
