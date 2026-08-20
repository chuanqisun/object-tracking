# Deploying YOLO26 Segmentation Turn-Key Inference Server on AMD XDNA 2

## Goal

This guide deploys a hardware-accelerated YOLO26 segmentation service on an AMD Strix Point/XDNA 2 NPU using a **turn-key, fully self-contained Docker container**.

The entire inferencing environment—including the AMD Ryzen AI SDK, static FP32 YOLO26 ONNX model, automatic BF16 Vitis AI Execution Provider runtime, and WebSocket server—is **baked into the container image at build time**. No manual file copying (`docker cp`), manual environment activation, or post-startup setup is required.

```text
Fedora 44 host
├── amdxdna kernel driver & firmware
├── /dev/accel/accel0
└── WebSocket Client (Host Script / HTML Page)
              │ (ws://localhost:8765)
              ▼
yolo26-npu:latest Turn-Key Container
├── Built-in Ryzen AI 1.8.0 SDK
├── Baked yolo26s-seg.onnx (Static FP32)
├── Baked vaiml_config.json & entrypoint.sh
├── Vitis AI EP / VAIML (BF16 NPU compilation)
├── NPU Cache (/workspace/npu_cache)
└── Asynchronous WebSocket Server (port 8765)
```

The Ubuntu container supplies AMD’s documented SDK environment and application server. Fedora supplies the host kernel driver and NPU device node `/dev/accel/accel0`.

**References:** [AMD Ryzen AI Linux documentation](https://ryzenai.docs.amd.com/en/latest/linux.html), [AMD XDNA driver](https://github.com/amd/xdna-driver)

---

# Phase 1: Configure the Fedora Host

## Step 1.1: Confirm Platform Requirements

Check the running kernel, PCI device, and kernel configuration:

```bash
uname -r

lspci -nn | grep -Ei \
    '1022:17f0|signal processing|coprocessor'

grep -E \
    'CONFIG_DRM_ACCEL=|CONFIG_AMD_IOMMU=' \
    /boot/config-$(uname -r)
```

Expected kernel configuration:

```text
CONFIG_DRM_ACCEL=y
CONFIG_AMD_IOMMU=y
```

---

## Step 1.2: Install host XDNA driver and XRT

Build and install the host `amdxdna` kernel module and XRT packages according to the official AMD driver repository. Verify module loading:

```bash
sudo modprobe amdxdna
lsmod | grep amdxdna
```

---

## Step 1.3: Configure device access and unlimited locked memory

Grant user permissions to hardware device nodes:

```bash
sudo usermod -aG render,video "$USER"
```

Configure unlimited locked memory for XRT mmap operations:

```bash
sudo tee /etc/security/limits.d/99-amdxdna.conf >/dev/null <<'EOF'
* soft memlock unlimited
* hard memlock unlimited
EOF
```

Apply the limits to systemd user services:

```bash
sudo mkdir -p /etc/systemd/user.conf.d

sudo tee /etc/systemd/user.conf.d/99-amdxdna.conf >/dev/null <<'EOF'
[Manager]
DefaultLimitMEMLOCK=infinity
EOF
```

Reboot host if applying limits for the first time:

```bash
sudo reboot
```

### Validation

```bash
ls -l /dev/accel/accel0
ulimit -l
```

Confirm `/dev/accel/accel0` exists and `ulimit -l` reports `unlimited`.

---

# Phase 2: Build the Turn-Key Docker Container

All required build assets must be placed inside `docker-data/`:
- `docker-data/ryzen_ai-1.8.0.tgz` (AMD Ryzen AI SDK archive)
- `docker-data/yolo26s-seg.pt` (PyTorch model weights)
- `docker-data/sample.jpg` (Sample test image)

## Step 2.1: Review the Dockerfile (`npu/Dockerfile`)

The container builds on top of `ryzerdocker:latest` and bakes all dependencies, models, and scripts into the image:

```dockerfile
FROM ryzerdocker:latest

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Copy Ryzen AI SDK archive and model assets from docker-data at build time
COPY docker-data/ryzen_ai-1.8.0.tgz /tmp/
COPY docker-data/yolo26s-seg.pt /workspace/yolo26s-seg.pt
COPY docker-data/sample.jpg /workspace/sample.jpg

# Install AMD Ryzen AI SDK inside the image
RUN mkdir -p /opt/ryzen-ai && \
    tar -xzf /tmp/ryzen_ai-1.8.0.tgz -C /opt/ryzen-ai && \
    rm -f /tmp/ryzen_ai-1.8.0.tgz && \
    cd /opt/ryzen-ai && \
    ./install_ryzen_ai.sh -a yes -p /opt/ryzen-ai/venv

# Activate venv and install required Python packages
RUN . /opt/ryzen-ai/venv/bin/activate && \
    pip install --no-cache-dir \
        "numpy<2" \
        "opencv-python-headless<4.11" \
        "opencv-python<4.11" \
        ultralytics \
        websockets \
        onnx

# Copy server application files
COPY npu/vaiml_config.json /workspace/vaiml_config.json
COPY npu/export_model.py /workspace/export_model.py
COPY npu/server.py /workspace/server.py
COPY npu/entrypoint.sh /workspace/entrypoint.sh

RUN chmod +x /workspace/entrypoint.sh

# Bake static FP32 ONNX model during build time
RUN . /opt/ryzen-ai/venv/bin/activate && \
    export PT_MODEL_PATH=/workspace/yolo26s-seg.pt && \
    export ONNX_MODEL_PATH=/workspace/yolo26s-seg.onnx && \
    python /workspace/export_model.py

EXPOSE 8765

ENTRYPOINT ["/workspace/entrypoint.sh"]
```

## Step 2.2: Build the Container Image

Run the build command from the workspace root:

```bash
docker build -t yolo26-npu:latest -f npu/Dockerfile .
```

---

# Phase 3: Run and Test the Turn-Key Inference Server

## Step 3.1: Run the Container

Launch the container with NPU device pass-through (`/dev/accel/accel0`) and unlimited locked memory (`--ulimit memlock=-1:-1`):

```bash
docker run -d \
  --name yolo26-npu-server \
  --device /dev/accel/accel0 \
  --ulimit memlock=-1:-1 \
  -p 8765:8765 \
  yolo26-npu:latest
```

Check the server logs to verify startup:

```bash
docker logs -f yolo26-npu-server
```

Expected startup logs:
```text
==================================================
 Starting Turn-Key YOLO26 NPU Inference Server
 Model Path:   /workspace/yolo26s-seg.onnx
 VAIML Config: /workspace/vaiml_config.json
 Listening:    ws://0.0.0.0:8765
==================================================
2026-08-19 22:30:00 [INFO] yolo26-npu-server: Initializing InferenceSession with VitisAIExecutionProvider...
2026-08-19 22:30:05 [INFO] yolo26-npu-server: Available Providers: ['VitisAIExecutionProvider', 'CPUExecutionProvider']
2026-08-19 22:30:05 [INFO] yolo26-npu-server: Warm-up run 1: 120.45 ms
2026-08-19 22:30:05 [INFO] yolo26-npu-server: Server is up and ready for connections on ws://0.0.0.0:8765
```

---

## Step 3.2: Test Inference from the Host

Run the client script from the host machine:

```bash
python npu/test_client.py
```

Expected output:
```text
Connecting to WebSocket server at ws://localhost:8765...
Connected! Sending requests...
[01/10] Detections=3, Inference=18.42ms, Server=21.15ms, RTT=23.50ms
[02/10] Detections=3, Inference=15.10ms, Server=17.20ms, RTT=18.90ms
...
--------------------------------------------------
Warm-up completed. Warm Median (p50) RTT: 18.90 ms
Warm 95th Percentile (p95) RTT:        21.30 ms
--------------------------------------------------
```

---

## Step 3.3: Observe NPU Activity

While requests are active, verify hardware execution inside the container:

```bash
docker exec -it yolo26-npu-server bash -c "source /opt/xilinx/xrt/setup.sh && xrt-smi examine"
```

The device report should show `RyzenAI-npu4` (`aie2p` architecture) present and actively executing inference workloads.

---

# Deployment Acceptance Checklist

## Host
- [x] AMD NPU appears in `lspci`.
- [x] `/dev/accel/accel0` exists and user is in `render`/`video` groups.
- [x] `amdxdna` kernel module is loaded.
- [x] Host `ulimit -l` reports `unlimited`.

## Container Image
- [x] Container image `yolo26-npu:latest` builds cleanly from `npu/Dockerfile`.
- [x] All assets (`ryzen_ai-1.8.0.tgz`, `yolo26s-seg.pt`, scripts) are copied from `docker-data/` at build time.
- [x] Static ONNX model (`yolo26s-seg.onnx`) is exported automatically at build time.

## Service Runtime
- [x] Container runs turn-key style via `docker run -d --device /dev/accel/accel0 --ulimit memlock=-1:-1 -p 8765:8765 yolo26-npu:latest`.
- [x] WebSocket server listens on `ws://0.0.0.0:8765`.
- [x] `VitisAIExecutionProvider` initializes successfully with NPU device access.
- [x] Host client receives valid detection bounding boxes, confidence scores, and segmentation contours.

