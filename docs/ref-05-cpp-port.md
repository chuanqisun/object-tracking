Here is an end-to-end blueprint showing how a **static web frontend** (pure HTML/JS/CSS served from GitHub Pages, S3/Cloudflare Pages, or opened locally as `index.html`) can communicate with a **standalone single-executable inference engine running CPU inference**.

---

# Architecture & Communication Design

Since the static frontend has no server-side logic of its own, it connects directly from the client's browser to the inference engine (listening on `ws://localhost:8000` or an IP/domain over WSS).

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ STATIC WEB FRONTEND (HTML5 / React / Vue / Plain JS)                                      │
│                                                                                           │
│  [navigator.mediaDevices.getUserMedia] ──► <video> (Display Stream)                       │
│                                                  │                                        │
│                                                  ▼ Frame Capture Loop                     │
│                                            Offscreen Canvas                               │
│                                                  │                                        │
│                                                  ▼ JPEG / Binary Buffer                   │
│                                              WebSocket                                    │
│                                                  │                                        │
│  <canvas> (Overlay Layer) ◄── Render Polygons ── JSON Responses                           │
└──────────────────────────────────┬────────────────────────────────────────────────────────┘
                                   │  Bidirectional WebSocket connection
                                   ▼  (ws://127.0.0.1:8000/ws)
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ STANDALONE INFERENCE ENGINE (Single Executable + Model File)                               │
│                                                                                           │
│  1. WebSocket Listener (Accepts binary JPEG/RGB bytes)                                    │
│  2. Fast Preprocessing (Resize & Normalize to Tensor)                                     │
│  3. Model Inference Engine (OpenVINO / ONNX Runtime C++ / Rust / Go)                       │
│  4. Fast Postprocessing (Boxes, Classes, Mask Contour Polygons)                           │
│  5. Response Dispatcher (Sends lightweight JSON payload)                                  │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Performance Strategy: Offload Mask Rendering

- **Old way (Streamlit):** CPU executes model $\rightarrow$ CPU generates alpha overlays onto image pixels via `res.plot()` $\rightarrow$ CPU re-encodes full frame $\rightarrow$ streams heavy image back.
- **Optimized way:** CPU executes model $\rightarrow$ extracts normalized polygon/box coordinates $\rightarrow$ sends lightweight JSON packet ($\approx 1\text{ KB}$) $\rightarrow$ Browser GPU/Canvas renders the segmentation overlays natively at 60 FPS.

---

# Detailed Single-Executable Engine Options

Below are the three concrete paths to build the inference executable, ranked by performance and runtime overhead.

---

### Option 1: Rust Single Binary + ONNX Runtime (Best Balance of Speed & Safety)

Rust compiles down to a single binary with zero external runtime dependencies. It supports automatic SIMD vectorization and multi-threaded WebSockets.

- **Tech Stack:**
  - **Model:** ONNX format (`puck-eye-seg-s.onnx`)
  - **Inference Library:** `ort` crate (safe Rust bindings to ONNX Runtime)
  - **Web Server:** `tokio` + `axum` / `tokio-tungstenite` for async WebSockets
  - **Image Decoding:** `image` crate (pure Rust JPEG/PNG decoding)
- **Output:** Single native binary (`inference-engine.exe` or `inference-engine`) $\approx 15\text{–}30\text{ MB}$.

#### Core Rust Engine Flow:

```rust
// Cargo.toml dependencies: ort, tokio, axum, serde_json, image
use axum::{extract::ws::{Message, WebSocket, WebSocketUpgrade}, routing::get, Router};
use ort::session::{Session, SessionOutputs};

async fn handle_ws(mut socket: WebSocket, session: Session) {
    while let Some(Ok(Message::Binary(img_bytes))) = socket.recv().await {
        // 1. Decode JPEG image directly from bytes
        let img = image::load_from_memory(&img_bytes).unwrap().to_rgb8();

        // 2. Preprocess: Resize & normalize to 1x3xHxW float tensor
        let tensor = preprocess_image(&img, 640, 640);

        // 3. Run CPU Inference
        let outputs: SessionOutputs = session.run(ort::inputs!["images" => tensor].unwrap()).unwrap();

        // 4. Extract boxes and mask contours
        let detections = postprocess_yolo_seg(&outputs);

        // 5. Send detections back as JSON text
        let json_payload = serde_json::to_string(&detections).unwrap();
        socket.send(Message::Text(json_payload)).await.unwrap();
    }
}
```

---

### Option 2: C++ Single Binary + OpenVINO (Maximum CPU Speed on x86/Intel)

OpenVINO provides the fastest CPU inference by leveraging hardware-specific vector instructions (AVX-512, AMX, VNNI).

- **Tech Stack:**
  - **Model:** OpenVINO IR (`model.xml` + `model.bin`) or `.onnx`
  - **Inference Library:** Intel OpenVINO C++ API
  - **Web/WebSocket Server:** `uWebSockets` (ultra-fast C++ WebSocket library) or `ixwebsocket`
  - **Image Processing:** OpenCV C++ or `stb_image`
- **Output:** Single compiled C++ executable + model file.

#### Core C++ Engine Flow:

```cpp
#include <openvino/openvino.hpp>
#include <uwebsockets/App.h>
#include <opencv2/opencv.hpp>

int main() {
    ov::Core core;
    // Optimize specifically for CPU low-latency throughput
    core.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    core.set_property("CPU", ov::hint::num_requests(1));
    auto compiled_model = core.compile_model("puck-eye-seg-s.xml", "CPU");
    auto infer_request = compiled_model.create_infer_request();

    uWS::App().ws<UserData>("/*", {
        .message = [&](auto *ws, std::string_view message, uWS::OpCode opCode) {
            if (opCode == uWS::OpCode::BINARY) {
                // Decode frame buffer from WebSocket
                std::vector<uchar> buffer(message.begin(), message.end());
                cv::Mat frame = cv::imdecode(buffer, cv::IMREAD_COLOR);

                // Run pre-allocated inference tensor
                run_inference(infer_request, frame);

                // Serialize mask polygons & bboxes to JSON
                std::string json_result = parse_seg_output(infer_request);
                ws->send(json_result, uWS::OpCode::TEXT);
            }
        }
    }).listen(8000, [](auto *listen_socket) {
        if (listen_socket) std::cout << "Engine listening on port 8000\n";
    }).run();
}
```

---

### Option 3: Python Standalone Bundle via PyInstaller (Fastest to Implement)

If you prefer to stay in Python without maintaining PyTorch in memory, you can run OpenVINO / ONNX Runtime via a lightweight **FastAPI WebSocket server** and compile it into a single binary with `PyInstaller`.

- **Tech Stack:**
  - **Model:** OpenVINO format or ONNX
  - **Inference Engine:** `openvino-telemetry` removed, pure `openvino` package (no `torch`)
  - **Web Server:** `FastAPI` + `uvicorn`
- **Packaging:** `pyinstaller --onefile server.py`
- **Output:** A single `.exe` or executable binary of $\approx 80\text{–}120\text{ MB}$.

#### `server.py`:

```python
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenVINO directly (does NOT load PyTorch)
model = YOLO("puck-eye-seg-s_openvino_model/", task="segment")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        bytes_data = await websocket.receive_bytes()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        results = model.predict(frame, conf=0.35, imgsz=480, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            polygons = results.masks.xy if results.masks is not None else []

            for i, box in enumerate(boxes):
                poly = polygons[i].tolist() if i < len(polygons) else []
                detections.append({
                    "class": model.names[int(classes[i])],
                    "conf": round(float(confs[i]), 2),
                    "box": [round(v, 1) for v in box.tolist()],
                    "polygon": poly
                })

        await websocket.send_json({"objects": detections})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

---

# Complete Static Web Frontend (`index.html`)

This single file contains everything needed: video capture, WebSocket streaming loop, and HTML5 Canvas rendering of bounding boxes and translucent segmentation masks.

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>YOLO Real-Time Segmentation Frontend</title>
    <style>
      body {
        font-family: sans-serif;
        background: #111;
        color: #eee;
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 0;
        padding: 20px;
      }
      .video-container {
        position: relative;
        width: 640px;
        height: 480px;
        margin-top: 15px;
        border: 2px solid #333;
        border-radius: 8px;
        overflow: hidden;
      }
      video,
      canvas {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
      }
      .controls {
        display: flex;
        gap: 12px;
        align-items: center;
      }
      button {
        padding: 8px 16px;
        background: #0070f3;
        border: none;
        color: #fff;
        font-weight: bold;
        border-radius: 4px;
        cursor: pointer;
      }
      button:disabled {
        background: #555;
      }
      #stats {
        font-family: monospace;
      }
    </style>
  </head>
  <body>
    <h1>Real-Time Segmentation Viewer</h1>

    <div class="controls">
      <button id="startBtn">Start Camera</button>
      <button id="stopBtn" disabled>Stop</button>
      <span id="stats">Engine FPS: 0 | Latency: 0ms</span>
    </div>

    <div class="video-container">
      <!-- Raw camera stream -->
      <video id="webcam" autoplay playsinline muted></video>
      <!-- Overlay canvas for bounding boxes & segmentation masks -->
      <canvas id="overlayCanvas" width="640" height="480"></canvas>
    </div>

    <!-- Offscreen canvas for grabbing frames as JPEG -->
    <canvas id="captureCanvas" width="640" height="480" style="display:none;"></canvas>

    <script>
      const video = document.getElementById("webcam");
      const overlayCanvas = document.getElementById("overlayCanvas");
      const ctx = overlayCanvas.getContext("2d");
      const captureCanvas = document.getElementById("captureCanvas");
      const captureCtx = captureCanvas.getContext("2d");
      const statsEl = document.getElementById("stats");
      const startBtn = document.getElementById("startBtn");
      const stopBtn = document.getElementById("stopBtn");

      let ws = null;
      let stream = null;
      let isRunning = false;
      let frameSendTime = 0;
      let isWaitingForResponse = false;

      async function initWebSocket() {
        return new Promise((resolve, reject) => {
          ws = new WebSocket("ws://127.0.0.1:8000/ws");
          ws.binaryType = "arraybuffer";

          ws.onopen = () => {
            console.log("Connected to Inference Engine");
            resolve();
          };

          ws.onmessage = (event) => {
            const latency = Date.now() - frameSendTime;
            const data = JSON.parse(event.data);
            drawResults(data.objects);
            statsEl.innerText = `Latency: ${latency}ms | Detected: ${data.objects.length}`;
            isWaitingForResponse = false; // Ready to send next frame
          };

          ws.onerror = (err) => reject(err);
          ws.onclose = () => {
            stopStream();
          };
        });
      }

      startBtn.onclick = async () => {
        try {
          await initWebSocket();
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
          });
          video.srcObject = stream;
          isRunning = true;
          startBtn.disabled = true;
          stopBtn.disabled = false;
          requestAnimationFrame(streamLoop);
        } catch (err) {
          alert("Failed to connect to inference engine or open webcam: " + err);
        }
      };

      stopBtn.onclick = () => stopStream();

      function stopStream() {
        isRunning = false;
        if (stream) stream.getTracks().forEach((t) => t.stop());
        if (ws) ws.close();
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        startBtn.disabled = false;
        stopBtn.disabled = true;
      }

      // Capture and stream loop with backpressure control
      function streamLoop() {
        if (!isRunning) return;

        if (!isWaitingForResponse && ws && ws.readyState === WebSocket.OPEN) {
          // Draw video frame to hidden canvas
          captureCtx.drawImage(video, 0, 0, 640, 480);

          // Export as JPEG blob and send to engine
          captureCanvas.toBlob(
            (blob) => {
              if (blob && ws.readyState === WebSocket.OPEN) {
                frameSendTime = Date.now();
                isWaitingForResponse = true;
                blob.arrayBuffer().then((buf) => ws.send(buf));
              }
            },
            "image/jpeg",
            0.7,
          ); // 0.7 quality is fast and sufficient for inference
        }

        requestAnimationFrame(streamLoop);
      }

      // Client-side GPU/Canvas rendering of segmentations
      function drawResults(objects) {
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        objects.forEach((obj) => {
          const [x1, y1, x2, y2] = obj.box;

          // 1. Draw segmentation mask polygon if present
          if (obj.polygon && obj.polygon.length > 0) {
            ctx.beginPath();
            ctx.moveTo(obj.polygon[0][0], obj.polygon[0][1]);
            for (let i = 1; i < obj.polygon.length; i++) {
              ctx.lineTo(obj.polygon[i][0], obj.polygon[i][1]);
            }
            ctx.closePath();
            ctx.fillStyle = "rgba(0, 255, 128, 0.35)";
            ctx.fill();
            ctx.lineWidth = 2;
            ctx.strokeStyle = "rgba(0, 255, 128, 0.9)";
            ctx.stroke();
          }

          // 2. Draw bounding box
          ctx.strokeStyle = "#00ffff";
          ctx.lineWidth = 2;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

          // 3. Draw text label
          const label = `${obj.class} ${Math.round(obj.conf * 100)}%`;
          ctx.fillStyle = "#00ffff";
          ctx.font = "14px sans-serif";
          ctx.fillText(label, x1, y1 > 15 ? y1 - 5 : y1 + 15);
        });
      }
    </script>
  </body>
</html>
```

---

# Summary Comparison

| Metric / Consideration        | Option 1: Rust + ONNX Runtime     | Option 2: C++ + OpenVINO          | Option 3: Python PyInstaller       |
| :---------------------------- | :-------------------------------- | :-------------------------------- | :--------------------------------- |
| **Max CPU Inference Speed**   | ⭐⭐⭐⭐ (Very Fast)              | ⭐⭐⭐⭐⭐ (Fastest, AVX-512)     | ⭐⭐⭐ (Good with OpenVINO)        |
| **RAM Footprint**             | $\approx 80\text{–}150\text{ MB}$ | $\approx 60\text{–}120\text{ MB}$ | $\approx 350\text{–}600\text{ MB}$ |
| **Executable Size**           | $\approx 25\text{ MB}$            | $\approx 15\text{–}30\text{ MB}$  | $\approx 90\text{–}140\text{ MB}$  |
| **Implementation Complexity** | Medium                            | Medium/High                       | Low (Quickest)                     |
| **Dependencies on User Host** | None (Single file)                | None (Single file)                | None (Single file)                 |
