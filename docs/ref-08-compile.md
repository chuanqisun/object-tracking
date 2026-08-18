To get the maximum frame rate and minimum latency out of your **AMD Ryzen AI 9 HX 370 (Zen 5 CPU + Radeon 890M RDNA 3.5 iGPU)** on Fedora Linux, moving away from PyTorch (`.pt`) and Python's GIL/overhead is the right move.

Here is the breakdown of why your current setup is slow, the best acceleration backends for your hardware, and a recommended architecture for a self-contained single-binary server.

---

### 1. Hardware & Runtime Acceleration Options for AMD 890M

| Runtime / Engine              | Target Hardware                           | Pros                                                                                                                                                         | Cons / Notes                                                                                            |
| :---------------------------- | :---------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| **ONNX Runtime (C++ / Rust)** | **CPU (AVX-512)** / **Vulkan** / **ROCm** | Most versatile, direct `.onnx` export from YOLO, easy C++/Rust API.                                                                                          | Native ROCm for RDNA 3.5 on Fedora can require specific environment flags (`HSA_OVERRIDE_GFX_VERSION`). |
| **NCNN (C++)**                | **Radeon 890M (Vulkan)**                  | **Fastest & most lightweight GPU inference** for consumer AMD iGPUs. Zero ROCm driver headache; works out of the box with standard RADV/Mesa Vulkan drivers. | Requires exporting YOLO to NCNN (`.param` / `.bin`).                                                    |
| **OpenVINO (C++)**            | **Zen 5 CPU (AVX-512/VNNI)**              | Highly optimized x86 CPU performance with automatic weight quantization (FP16/INT8).                                                                         | Targets CPU primarily on AMD systems (Intel GPU only for graphics acceleration).                        |

---

### 2. The Recommended Stack

For a **fast, standalone single executable + single model file**:

1. **Export Format:** Export your YOLO model to **`ONNX`** (or **`NCNN`**).
   ```bash
   # From your Python environment
   yolo export model=yolo-seg.pt format=onnx imgsz=640 half=True dynamic=False
   # Or for NCNN
   yolo export model=yolo-seg.pt format=ncnn imgsz=640 half=True
   ```
2. **Inference Engine in Binary:**
   - **Option A (GPU via Vulkan - Lowest Latency / Best FPS):** **NCNN** or **ONNX Runtime (Execution Provider: CPU with AVX-512 or DirectML/ROCm/Vulkan)**.
   - **Option B (CPU via AVX-512 - Easiest & extremely fast on Zen 5):** **ONNX Runtime C++ API** (with OpenMP & AVX-512 enabled). The 12 Zen 5 cores on the HX 370 can process standard 640x640 YOLO segmentation in ~6–10 ms on CPU alone.
3. **Web / WebSocket Server:**
   - **C++:** [uWebSockets](https://github.com/uNetworking/uWebSockets) (one of the fastest WebSocket libraries in existence) or [CrowCpp](https://crowcpp.org/).
   - **Rust:** [Actix-web](https://actix.rs/) / [Axum](https://github.com/tokio-rs/axum) + `ort` (ONNX Runtime Rust bindings).

---

### 3. Architecture for a Self-Contained Server

```
┌────────────────────────┐       WebSocket (Binary / JSON)       ┌────────────────────────────────────────────────────────┐
│                        │ ◄───────────────────────────────────► │ Single Native Executable                              │
│ Frontend (Browser/App) │                                       │ ├─ uWebSockets / Axum Server                           │
│                        │                                       │ ├─ Pre-processing (OpenCV / fast resize / normalize)   │
└────────────────────────┘                                       │ ├─ ONNXRuntime / NCNN Engine (loads model.onnx)        │
                                                                 │ └─ Post-processing (NMS + Mask Decode via SIMD/OpenMP)│
                                                                 └────────────────────────────────────────────────────────┘
```

---

### 4. Step-by-Step Implementation Guide

#### Option A: High-Performance Rust Server (Easiest to build & statically ship)

Rust provides clean concurrency, zero-cost memory safety, and easy static compilation into a single binary.

1. **Setup Dependencies (`Cargo.toml`):**

   ```toml
   [dependencies]
   tokio = { version = "1", features = ["full"] }
   axum = { version = "0.7", features = ["ws"] }
   ort = { version = "2.0.0-rc.9" } # ONNX Runtime bindings
   image = "0.25"
   ndarray = "0.16"
   serde_json = "1.0"
   ```

2. **Inference & WebSocket Pipeline:**
   - Initialize the `ort::Session` once at startup loading `model.onnx`.
   - On incoming WebSocket frame:
     - Decode image buffer.
     - Letterbox resize & normalize to `[1, 3, 640, 640]` float32 tensor.
     - Run `session.run(...)`.
     - Extract segmentation prototypes and bounding boxes, apply Non-Maximum Suppression (NMS).
     - Send detections & compressed mask polygon/RLE back over WebSocket as binary/JSON.

#### Option B: C++ with `uWebSockets` and `NCNN` (Maximum GPU Utilization via Vulkan)

Because Fedora ships with high-performance Mesa RADV drivers for the Radeon 890M:

1. **Build NCNN with Vulkan support:**
   ```bash
   git clone https://github.com/Tencent/ncnn.git
   cd ncnn && mkdir build && cd build
   cmake -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=OFF ..
   make -j$(nproc)
   ```
2. Integrate `ncnn::Extractor` inside a `uWebSockets` message loop.
3. NCNN automatically utilizes the 16 Compute Units of the Radeon 890M over Vulkan compute pipelines with zero heavy CUDA/ROCm daemon requirements.

---

### 5. Quick Wins to Boost Speed Immediately

- **FP16 / Half Precision:** Ensure you export the model in `half=True` (FP16). Both Zen 5 CPU (via AVX-512 FP16) and Radeon 890M have dedicated FP16 dual-issue tensor compute.
- **Fixed Input Resolution:** Avoid dynamic shapes (`dynamic=False`) to allow the engine to pre-allocate memory buffers and skip runtime shape reallocation.
- **Batch = 1 / Direct Memory:** Allocate pre-allocated input/output buffers so image decoding writes directly into the inference tensor memory.
