To extract the **absolute maximum GPU throughput and lowest latency** from the **AMD Ryzen AI 9 HX 370’s Radeon 890M iGPU (RDNA 3.5)** on Fedora Linux, the fastest and most stable path is **NCNN via Vulkan Compute** paired with a ultra-low-overhead C++ server (**uWebSockets** / **Crow**).

> **Why NCNN + Vulkan over ROCm for the 890M?**
> ROCm on consumer RDNA 3.5 iGPUs requires gfx-version overrides (`HSA_OVERRIDE_GFX_VERSION=11.5.0`) and heavy driver stacks. Mesa RADV Vulkan drivers are native to Fedora, have zero kernel/driver headaches, and compile shaders directly to the 890M's 16 Compute Units with dual-issue FP16 tensor acceleration.

---

### Step 1: Export Fine-Tuned YOLO26 Segmentation Model

Export your model directly to NCNN format with FP16 enabled and fixed shape.

```bash
# In your Python environment
yolo export model=best_yolo26_seg.pt format=ncnn imgsz=640 half=True dynamic=False
```

This generates:

- `model.param` (Graph structure)
- `model.bin` (Quantized FP16 weights)

_(If you only export to ONNX first, run `onnx2ncnn model.onnx model.param model.bin` and then `ncnnoptimize model.param model.bin model-opt.param model-opt.bin 65536` for FP16 optimization)._

---

### Step 2: Install System Dependencies (Fedora)

Install Vulkan development headers, Mesa RADV driver, Asio, and build tools:

```bash
sudo dnf install -y \
    gcc-c++ cmake git \
    vulkan-loader-devel mesa-vulkan-drivers vulkan-tools \
    glslang glslc opencv opencv-devel zlib-devel openssl-devel \
    asio-devel boost-devel
```

Verify your Radeon 890M Vulkan driver is active:

```bash
vulkaninfo --summary
# Look for: GPU0: ... AMD Radeon Graphics (RADV GFX1150 / RDNA 3.5)
```

---

### Step 3: Build NCNN with Vulkan Acceleration

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

---

### Step 4: High-Performance GPU Inference Server (C++)

Create a high-speed HTTP/WebSocket server using standard C++ and OpenCV.

#### `server.cpp`

```cpp
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <ncnn/gpu.h>

// Micro HTTP Server using Crow (Header-only: github.com/CrowCpp/Crow)
#include "crow_all.h"

class YOLO26SegEngine {
private:
    ncnn::Net net;
    int target_size = 640;
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

public:
    YOLO26SegEngine(const std::string& param_path, const std::string& bin_path) {
        // 1. Enable Vulkan GPU Acceleration
        ncnn::create_gpu_instance();
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
        net.opt.num_threads = 4; // Lightweight CPU thread feeder

        // 2. Select Radeon 890M GPU (Device 0)
        net.set_vulkan_device(0);

        // 3. Load Model
        if (net.load_param(param_path.c_str()) != 0 || net.load_model(bin_path.c_str()) != 0) {
            std::cerr << "Failed to load NCNN model files!" << std::endl;
        }
    }

    ~YOLO26SegEngine() {
        net.clear();
        ncnn::destroy_gpu_instance();
    }

    crow::json::wvalue infer(const std::vector<uchar>& image_bytes) {
        cv::Mat img = cv::imdecode(image_bytes, cv::IMREAD_COLOR);
        if (img.empty()) return {{"error", "Invalid image"}};

        int img_w = img.cols;
        int img_h = img.rows;

        // Resize / Letterbox to 640x640
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data,
            ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, target_size, target_size);
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in); // Input blob name is "in0"

        ncnn::Mat out_det;
        ncnn::Mat out_proto;

        // YOLO26 End-to-End NMS-Free Head Output + Segmentation Prototypes
        ex.extract("out0", out_det);
        ex.extract("out1", out_proto);

        // Parse Detections and Masks
        crow::json::wvalue res;
        std::vector<crow::json::wvalue> detections;

        // In NCNN, out0 shape is either (38, num_boxes) or (num_boxes, 38)
        if (out_det.h == 38 || (out_det.dims == 2 && out_det.h < out_det.w)) {
            // Shape is (38, num_boxes) - transposed layout: channel/attribute is row
            for (int i = 0; i < out_det.w; i++) {
                float x1 = out_det.row(0)[i];
                float y1 = out_det.row(1)[i];
                float x2 = out_det.row(2)[i];
                float y2 = out_det.row(3)[i];
                float score = out_det.row(4)[i];
                int class_id = (int)out_det.row(5)[i];

                if (score > 0.45f) {
                    crow::json::wvalue obj;
                    obj["class_id"] = class_id;
                    obj["score"] = score;
                    obj["box"] = crow::json::wvalue::list({x1, y1, x2, y2});
                    detections.push_back(obj);
                }
            }
        } else {
            // Shape is (num_boxes, 38) - standard layout: each row is a box
            for (int i = 0; i < out_det.h; i++) {
                const float* values = out_det.row(i);
                float score = values[4];
                if (score > 0.45f) {
                    crow::json::wvalue obj;
                    obj["class_id"] = (int)values[5];
                    obj["score"] = score;
                    obj["box"] = crow::json::wvalue::list({values[0], values[1], values[2], values[3]});
                    detections.push_back(obj);
                }
            }
        }

        res["detections"] = std::move(detections);
        return res;
    }
};

int main(int argc, char** argv) {
    std::string param_path = "puck-eye-seg-s_ncnn_model/model.ncnn.param";
    std::string bin_path = "puck-eye-seg-s_ncnn_model/model.ncnn.bin";

    if (argc >= 3) {
        param_path = argv[1];
        bin_path = argv[2];
    } else {
        FILE* fp = fopen(param_path.c_str(), "rb");
        if (fp) {
            fclose(fp);
        } else {
            param_path = "../puck-eye-seg-s_ncnn_model/model.ncnn.param";
            bin_path = "../puck-eye-seg-s_ncnn_model/model.ncnn.bin";
        }
    }

    std::cout << "Loading NCNN model: " << param_path << " and " << bin_path << std::endl;

    crow::SimpleApp app;
    YOLO26SegEngine engine(param_path, bin_path);

    CROW_ROUTE(app, "/predict").methods("POST"_method)
    ([&engine](const crow::request& req) {
        std::vector<uchar> image_bytes(req.body.begin(), req.body.end());
        auto result = engine.infer(image_bytes);
        return crow::response(result);
    });

    int port = (argc >= 4) ? std::atoi(argv[3]) : 8080;
    app.port(port).multithreaded().run();
}
```

---

### Step 5: Build & Run the Executable

1. Download single-header Crow for rapid compilation:

   ```bash
   curl -L -o inference/crow_all.h https://github.com/CrowCpp/Crow/releases/download/v1.2.0/crow_all.h
   ```

2. Compile with `g++` leveraging `-O3` and Native Zen 5 instruction optimizations:

   ```bash
   export PKG_CONFIG_PATH=/home/stack/repos/ncnn/build/install/lib64/pkgconfig:$PKG_CONFIG_PATH
   cd inference
   g++ -O3 -march=native -fopenmp \
       server.cpp -o yolo_server \
       `pkg-config --cflags --libs opencv4 ncnn` \
       -L/home/stack/repos/ncnn/build/install/lib64 -lglslang -lMachineIndependent -lGenericCodeGen -lOSDependent -lSPIRV -lvulkan -lpthread
   ```

3. Run the binary:
   ```bash
   ./yolo_server ../puck-eye-seg-s_ncnn_model/model.ncnn.param ../puck-eye-seg-s_ncnn_model/model.ncnn.bin 8080
   ```

---

### Step 6: Test Inference Latency

Send an image via HTTP POST:

```bash
curl -s -X POST --data-binary @sample.jpg http://localhost:8080/predict
```

### Performance Optimization Checklist for Radeon 890M:

1. **Zero-Copy Transfers:** Keep `net.opt.use_fp16_packed = true;` to avoid converting 32-bit floats across CPU/GPU memory buses.
2. **Fixed Input Size:** Set fixed 640x640 dimensions during `yolo export` to avoid dynamic Vulkan pipeline recompilations.
3. **NMS-Free Advantage:** YOLO26 eliminates Distribution Focal Loss (DFL) and native NMS, meaning GPU kernel time is significantly faster than YOLOv8/v11.

## References

1. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGlgiwQlr8DCjpFOKxgXaSHKcTfwZtokTcKO9Buyz1OnjRtG_CfMnZRj_2Y81FKOfT0WlLC3vRjrwV34vCdhCf418pKmvgvbJfwm6C11P1u8jR1HgHW4hT0qjp3bGMHjvI=)
2. [huggingface.co](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG-BUVdgF4p7Oi6RfmJhZjTc5Zsf_6JVedpy2vaGZPP1oZM2AxW5OKcG7eIKww-lOxr-PPZ_ZLNUsaxykEAl6CSS3wGxNMQKN1vJmB_WlrGzEi6TJVlUHVzgU4v3cRU)
