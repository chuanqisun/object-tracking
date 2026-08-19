# Deploying YOLO26 Segmentation with Automatic BF16 Compilation on AMD XDNA 2

## Goal

This guide deploys a hardware-accelerated YOLO26 segmentation service on an AMD Strix Point/XDNA 2 NPU.

The final system consists of:

- Fedora 44 host running the `amdxdna` kernel driver
- Ubuntu 24.04 container with AMD Ryzen AI software
- Static FP32 YOLO26 segmentation ONNX model
- Automatic BF16 compilation through the Vitis AI Execution Provider
- Asynchronous WebSocket server returning boxes and segmentation contours
- Host-side validation of accuracy, latency, and NPU execution

```text
Fedora 44 host
├── amdxdna kernel driver
├── matching XRT plug-in and firmware
├── /dev/accel/accel0
└── WebSocket client
              │
              ▼
Ubuntu 24.04 container
├── matching XRT userspace
├── Ryzen AI SDK
├── FP32 YOLO26-seg ONNX model
├── Vitis AI EP / VAIML
│   ├── supported graph partitions → BF16 NPU execution
│   └── remaining graph partitions → CPU execution
├── compiled-model cache
└── asynchronous WebSocket server
```

The Ubuntu container supplies AMD’s documented SDK environment. Fedora supplies the kernel driver and NPU device node.

**References:** [AMD Ryzen AI Linux documentation](https://ryzenai.docs.amd.com/en/latest/linux.html), [AMD XDNA driver](https://github.com/amd/xdna-driver)

---

# Phase 1: Configure the Fedora host

## Step 1.1: Confirm the platform

Check the running kernel, PCI device and required kernel configuration:

```bash
uname -r

lspci -nn | grep -Ei \
    '1022:17f0|signal processing|coprocessor'

grep -E \
    'CONFIG_DRM_ACCEL=|CONFIG_AMD_IOMMU=' \
    /boot/config-$(uname -r)
```

Expected kernel settings:

```text
CONFIG_DRM_ACCEL=y
CONFIG_AMD_IOMMU=y
```

### Validation

Confirm that:

- The AMD NPU appears in `lspci`.
- `CONFIG_DRM_ACCEL` is enabled.
- `CONFIG_AMD_IOMMU` is enabled.
- The running kernel is compatible with the current XDNA driver.

**References:** [AMD XDNA driver requirements](https://github.com/amd/xdna-driver)

---

## Step 1.2: Build and install the complete XDNA stack

Install the build dependencies:

```bash
sudo dnf group install -y "Development Tools"

sudo dnf install -y \
    git git-lfs dkms cmake ninja-build \
    kernel-devel-$(uname -r) \
    kernel-headers-$(uname -r) \
    libdrm-devel boost-devel rpm-build
```

Clone the driver repository and initialize its submodules:

```bash
git clone https://github.com/amd/xdna-driver.git
cd xdna-driver

git submodule update --init --recursive
sudo ./tools/amdxdna_deps.sh
```

Build XRT:

```bash
cd xrt/build
./build.sh -npu -opt
```

Build the XDNA release packages:

```bash
cd ../../build
./build.sh -release
```

Install the generated packages:

```bash
sudo dnf install -y ./Release/xrt-*.rpm
sudo dnf install -y ./Release/xrt_plugin-*.rpm

sudo modprobe amdxdna
```

### Validation

```bash
modinfo amdxdna
lsmod | grep amdxdna
```

Both commands should identify the installed `amdxdna` module.

**References:** [AMD XDNA driver build instructions](https://github.com/amd/xdna-driver)

---

## Step 1.3: Configure device access and locked memory

Add the current user to the hardware-access groups:

```bash
sudo usermod -aG render,video "$USER"
```

Configure unlimited locked memory:

```bash
sudo tee /etc/security/limits.d/99-amdxdna.conf >/dev/null <<'EOF'
* soft memlock unlimited
* hard memlock unlimited
EOF
```

Apply the same limit to systemd user services:

```bash
sudo mkdir -p /etc/systemd/user.conf.d

sudo tee /etc/systemd/user.conf.d/99-amdxdna.conf >/dev/null <<'EOF'
[Manager]
DefaultLimitMEMLOCK=infinity
EOF
```

Reboot:

```bash
sudo reboot
```

### Validation

After logging in again:

```bash
id
ulimit -l
```

Confirm that:

- `render` and `video` appear in the group list.
- The locked-memory limit reports `unlimited`.

**References:** [AMD XDNA driver documentation](https://github.com/amd/xdna-driver)

---

## Step 1.4: Validate the host NPU

Initialize the XRT environment:

```bash
source /opt/xilinx/xrt/setup.sh
```

Inspect the device:

```bash
ls -l /dev/accel/
lsmod | grep amdxdna

xrt-smi examine
xrt-smi validate
```

### Validation

Confirm that:

- `/dev/accel/accel0` exists.
- The current user can access the device.
- `xrt-smi examine` recognizes the AMD NPU.
- The reported architecture is `aie2p`.
- Supported validation tests finish with `[PASSED]`.

The exact device name and array topology can vary by processor.

**References:** [AMD Ryzen AI Linux installation and validation](https://ryzenai.docs.amd.com/en/latest/linux.html)

---

# Phase 2: Create the container environment

## Step 2.1: Install Ryzers

Clone Ryzers and install it into a virtual environment:

```bash
git clone https://github.com/AMDResearch/Ryzers.git
cd Ryzers

python3 -m venv ~/.venvs/ryzers
source ~/.venvs/ryzers/bin/activate

pip install --upgrade pip
pip install -e .
```

Verify the command-line interface:

```bash
ryzers --help
```

### Validation

The command should display the Ryzers build and run options.

**References:** [AMDResearch Ryzers](https://github.com/AMDResearch/Ryzers)

---

## Step 2.2: Build and test the XDNA container

Build the base image:

```bash
cd ~/Ryzers
ryzers build xdna
```

Start the container:

```bash
ryzers run --name xdna
```

Inside the container:

```bash
source /opt/xilinx/xrt/setup.sh

ls -l /dev/accel/
xrt-smi examine
xrt-smi validate
```

### Validation

Confirm that:

- `/dev/accel/accel0` is visible.
- The NPU information matches the host.
- XRT validation passes inside the container.

For this NPU-only service, `/dev/accel/accel0` is the required device. GPU device nodes are only needed if the application also uses the GPU.

**References:** [Ryzers XDNA package](https://github.com/AMDResearch/Ryzers/tree/main/packages/npu/xdna)

---

## Step 2.3: Install the Ryzen AI SDK

Obtain AMD’s current Linux Ryzen AI SDK archive and make it available inside the container.

Extract and install it:

```bash
mkdir -p /opt/ryzen-ai

tar -xzf /tmp/ryzen_ai-1.8.0.tgz \
    -C /opt/ryzen-ai

cd /opt/ryzen-ai

./install_ryzen_ai.sh \
    -a yes \
    -p /opt/ryzen-ai/venv
```

Activate the SDK environment:

```bash
source /opt/ryzen-ai/venv/bin/activate
source /opt/xilinx/xrt/setup.sh
```

Install the application dependencies:

```bash
pip install \
    ultralytics \
    websockets \
    opencv-python-headless \
    onnx
```

Preserve the ONNX Runtime package supplied by AMD because it contains the Vitis AI Execution Provider.

### Validation

```bash
python - <<'PY'
import onnxruntime as ort

print("ONNX Runtime:", ort.__version__)
print("Providers:", ort.get_available_providers())

assert "VitisAIExecutionProvider" in ort.get_available_providers()
PY
```

Expected provider list:

```text
VitisAIExecutionProvider
CPUExecutionProvider
```

**References:** [AMD Ryzen AI Linux installation](https://ryzenai.docs.amd.com/en/latest/linux.html)

---

## Step 2.4: Run AMD’s NPU quick test

Locate the test:

```bash
find /opt/ryzen-ai/venv \
    -path '*/quicktest/quicktest.py' \
    -print
```

Run it from the directory found above:

```bash
cd /opt/ryzen-ai/venv/quicktest
python quicktest.py
```

### Validation

Expected output includes:

```text
Setting environment for STX/KRK
Test Finished
```

This confirms that AMD’s runtime can compile and execute a test network on the NPU.

**References:** [AMD Ryzen AI quick-test instructions](https://ryzenai.docs.amd.com/en/latest/linux.html)

---

# Phase 3: Export the model

## Step 3.1: Export static FP32 ONNX

Create `export_model.py`:

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

result = model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    simplify=True,
    opset=17,
    end2end=True,
    half=False,
)

print(result)
```

Run the export:

```bash
python export_model.py
```

The resulting model should be:

```text
yolo26n-seg.onnx
```

The model remains FP32 in storage. VAIML selects BF16 execution for compatible NPU graph partitions during compilation.

### Validation

```bash
ls -lh yolo26n-seg.onnx
```

Confirm that the file exists and has a nonzero size.

**References:** [Ultralytics segmentation export](https://github.com/ultralytics/ultralytics/blob/main/docs/en/tasks/segment.md), [AMD Ryzen AI model compilation](https://ryzenai.docs.amd.com/en/latest/modelrun.html)

---

## Step 3.2: Inspect the ONNX graph

Create `inspect_model.py`:

```python
import onnx
import onnxruntime as ort

MODEL = "yolo26n-seg.onnx"

graph = onnx.load(MODEL)
onnx.checker.check_model(graph)

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"],
)

print("Inputs:")
for item in session.get_inputs():
    print(item.name, item.shape, item.type)

print("Outputs:")
for item in session.get_outputs():
    print(item.name, item.shape, item.type)

input_info = session.get_inputs()[0]

assert input_info.shape == [1, 3, 640, 640]
assert input_info.type == "tensor(float)"
```

Run it:

```bash
python inspect_model.py
```

Expected model layout:

```text
Input:      [1, 3, 640, 640]
Detections: [1, 300, 38]
Prototypes: [1, 32, 160, 160]
```

Each detection contains:

```text
x1, y1, x2, y2
confidence
class_id
32 mask coefficients
```

### Validation

Stop deployment and update the post-processing code if the exported output shapes differ from these values.

**References:** [Ultralytics end-to-end inference](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/end2end-detection.md)

---

## Step 3.3: Generate an FP32 reference

Create `make_reference.py`:

```python
import json
import cv2
import numpy as np
import onnxruntime as ort

MODEL = "yolo26n-seg.onnx"
IMAGE = "sample.jpg"
SIZE = 640

image = cv2.imread(IMAGE)
if image is None:
    raise RuntimeError(f"Unable to load {IMAGE}")

resized = cv2.resize(image, (SIZE, SIZE))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

tensor = (
    rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
)[None]

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"],
)

outputs = session.run(
    None,
    {session.get_inputs()[0].name: tensor},
)

np.savez(
    "fp32_reference.npz",
    output_0=outputs[0],
    output_1=outputs[1],
)

metadata = {
    "model": MODEL,
    "image": IMAGE,
    "output_shapes": [list(value.shape) for value in outputs],
    "output_dtypes": [str(value.dtype) for value in outputs],
}

with open("fp32_reference.json", "w") as file:
    json.dump(metadata, file, indent=2)

print(metadata)
```

Run it:

```bash
python make_reference.py
```

### Validation

Confirm that these files are generated:

```bash
ls -lh fp32_reference.npz fp32_reference.json
cat fp32_reference.json
```

**References:** [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)

---

# Phase 4: Compile the model for BF16 execution

## Step 4.1: Create the VAIML configuration

Create `vaiml_config.json`:

```json
{
  "passes": [
    {
      "name": "init",
      "plugin": "vaip-pass_init"
    },
    {
      "name": "vaiml_partition",
      "plugin": "vaip-pass_vaiml_partition",
      "vaiml_config": {
        "optimize_level": 1,
        "preferred_data_storage": "auto"
      }
    }
  ],
  "target": "VAIML",
  "targets": [
    {
      "name": "VAIML",
      "pass": ["init", "vaiml_partition"]
    }
  ]
}
```

Optimization level 1 provides the initial accuracy baseline.

### Validation

```bash
python -m json.tool vaiml_config.json >/dev/null
echo $?
```

An exit status of `0` confirms valid JSON syntax.

> Use the VAIML configuration shipped with the installed SDK if its schema differs from this example. Compiler configuration files are version-sensitive.

**References:** [AMD Ryzen AI model compilation](https://ryzenai.docs.amd.com/en/latest/modelrun.html)

---

## Step 4.2: Compile and cache the model

Create `compile_bf16.py`:

```python
import os
import time

import numpy as np
import onnxruntime as ort

MODEL = "yolo26n-seg.onnx"
CONFIG = "vaiml_config.json"
CACHE_DIR = "./npu_cache"
CACHE_KEY = "yolo26n_seg_fp32_auto_bf16_o1"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("./profiles", exist_ok=True)

options = ort.SessionOptions()
options.log_severity_level = 1
options.enable_profiling = True
options.profile_file_prefix = "./profiles/compile_bf16"

vai_options = {
    "config_file": os.path.abspath(CONFIG),
    "cache_dir": os.path.abspath(CACHE_DIR),
    "cache_key": CACHE_KEY,
    "enable_cache_file_io_in_mem": "0",
    "ai_analyzer_visualization": "true",
    "ai_analyzer_profiling": "true",
}

start = time.perf_counter()

session = ort.InferenceSession(
    MODEL,
    sess_options=options,
    providers=[
        "VitisAIExecutionProvider",
        "CPUExecutionProvider",
    ],
    provider_options=[
        vai_options,
        {},
    ],
)

print(
    "Session creation:",
    time.perf_counter() - start,
    "seconds",
)
print("Providers:", session.get_providers())
print(
    "Outputs:",
    [
        (output.name, output.shape, output.type)
        for output in session.get_outputs()
    ],
)

input_info = session.get_inputs()[0]
dummy = np.zeros(input_info.shape, dtype=np.float32)

for index in range(3):
    start = time.perf_counter()

    outputs = session.run(
        None,
        {input_info.name: dummy},
    )

    assert all(np.isfinite(value).all() for value in outputs)

    print(
        f"Warm-up {index}:",
        (time.perf_counter() - start) * 1000,
        "ms",
    )

print("Profile:", session.end_profiling())
```

Compile the model:

```bash
mkdir -p profiles npu_cache

source /opt/ryzen-ai/venv/bin/activate
source /opt/xilinx/xrt/setup.sh

python compile_bf16.py
```

### Validation

Confirm that:

- Session creation completes successfully.
- The provider list contains `VitisAIExecutionProvider`.
- All outputs contain finite values.
- Files appear in `npu_cache`.
- A profile is written under `profiles`.

```bash
find npu_cache -type f -ls
find profiles -type f -ls
```

**References:** [AMD Ryzen AI model compilation and caching](https://ryzenai.docs.amd.com/en/latest/modelrun.html)

---

## Step 4.3: Compare BF16 and FP32 outputs

Create `compare_bf16.py`:

```python
import os

import cv2
import numpy as np
import onnxruntime as ort

MODEL = "yolo26n-seg.onnx"
CONFIG = "vaiml_config.json"
IMAGE = "sample.jpg"
SIZE = 640

image = cv2.imread(IMAGE)
if image is None:
    raise RuntimeError(f"Unable to load {IMAGE}")

resized = cv2.resize(image, (SIZE, SIZE))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

tensor = (
    rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
)[None]

cpu = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"],
)

npu = ort.InferenceSession(
    MODEL,
    providers=[
        "VitisAIExecutionProvider",
        "CPUExecutionProvider",
    ],
    provider_options=[
        {
            "config_file": os.path.abspath(CONFIG),
            "cache_dir": os.path.abspath("./npu_cache"),
            "cache_key": "yolo26n_seg_fp32_auto_bf16_o1",
            "enable_cache_file_io_in_mem": "0",
        },
        {},
    ],
)

cpu_outputs = cpu.run(
    None,
    {cpu.get_inputs()[0].name: tensor},
)

npu_outputs = npu.run(
    None,
    {npu.get_inputs()[0].name: tensor},
)

for index, (reference, candidate) in enumerate(
    zip(cpu_outputs, npu_outputs)
):
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"Output {index} shape mismatch: "
            f"{reference.shape} vs {candidate.shape}"
        )

    if not np.isfinite(candidate).all():
        raise RuntimeError(
            f"Output {index} contains NaN or infinity"
        )

    difference = np.abs(reference - candidate)

    cosine = np.dot(
        reference.ravel(),
        candidate.ravel(),
    ) / (
        np.linalg.norm(reference.ravel())
        * np.linalg.norm(candidate.ravel())
        + 1e-12
    )

    print(f"Output {index}")
    print(f"  Shape: {reference.shape}")
    print(f"  Mean absolute error: {difference.mean()}")
    print(f"  Maximum absolute error: {difference.max()}")
    print(f"  Cosine similarity: {cosine}")
```

Run:

```bash
python compare_bf16.py
```

### Validation

Confirm that:

- Output shapes match.
- No outputs contain NaN or infinity.
- Detection and prototype tensors remain numerically close.
- Boxes, classes and segmentation masks remain correct on representative images.

Final acceptance should use dataset-level box and mask metrics rather than a single tensor tolerance.

**References:** [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/)

---

# Phase 5: Create the WebSocket service

## Step 5.1: Save the server

Create `server.py`:

```python
import asyncio
import base64
import json
import logging
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import websockets

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("yolo26-bf16")

SIZE = 640
CONFIDENCE = 0.40


def letterbox(image):
    height, width = image.shape[:2]
    gain = min(SIZE / height, SIZE / width)

    new_width = round(width * gain)
    new_height = round(height * gain)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)

    left = (SIZE - new_width) // 2
    top = (SIZE - new_height) // 2

    canvas[
        top:top + new_height,
        left:left + new_width,
    ] = resized

    return canvas, gain, left, top


class BF16Segmenter:
    def __init__(self):
        model = os.getenv(
            "MODEL_PATH",
            "/workspace/yolo26n-seg.onnx",
        )
        config = os.getenv(
            "VAIML_CONFIG",
            "/workspace/vaiml_config.json",
        )

        session_options = ort.SessionOptions()
        session_options.log_severity_level = int(
            os.getenv("ORT_LOG_LEVEL", "2")
        )

        if os.getenv("ORT_PROFILE", "0") == "1":
            session_options.enable_profiling = True
            session_options.profile_file_prefix = (
                "/workspace/profiles/yolo26_bf16"
            )

        vai_options = {
            "config_file": config,
            "cache_dir": "/workspace/npu_cache",
            "cache_key": "yolo26n_seg_fp32_auto_bf16_o1",
            "enable_cache_file_io_in_mem": "0",
        }

        if os.getenv("AI_ANALYZER", "0") == "1":
            vai_options.update({
                "ai_analyzer_visualization": "true",
                "ai_analyzer_profiling": "true",
            })

        self.session = ort.InferenceSession(
            model,
            sess_options=session_options,
            providers=[
                "VitisAIExecutionProvider",
                "CPUExecutionProvider",
            ],
            provider_options=[
                vai_options,
                {},
            ],
        )

        self.input = self.session.get_inputs()[0]
        self.outputs = self.session.get_outputs()

        if self.input.shape != [1, 3, 640, 640]:
            raise RuntimeError(
                f"Unexpected input shape: {self.input.shape}"
            )

        log.info("Providers: %s", self.session.get_providers())
        log.info(
            "Outputs: %s",
            [(item.name, item.shape) for item in self.outputs],
        )

        self.warmup()

    def warmup(self):
        dummy = np.zeros(
            (1, 3, SIZE, SIZE),
            dtype=np.float32,
        )

        for _ in range(3):
            self.session.run(
                None,
                {self.input.name: dummy},
            )

    def preprocess(self, image):
        padded, gain, left, top = letterbox(image)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        tensor = (
            rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        )[None]

        return tensor, gain, left, top

    @staticmethod
    def restore_boxes(boxes, gain, left, top, shape):
        boxes = boxes.copy()

        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes /= gain

        height, width = shape[:2]

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

        return boxes

    @staticmethod
    def create_polygons(
        prototypes,
        coefficients,
        boxes_640,
        original_shape,
        gain,
        left,
        top,
    ):
        if len(coefficients) == 0:
            return []

        channels, mask_height, mask_width = prototypes.shape

        logits = coefficients @ prototypes.reshape(channels, -1)
        masks = logits.reshape(-1, mask_height, mask_width)

        original_height, original_width = original_shape[:2]
        valid_height = round(original_height * gain)
        valid_width = round(original_width * gain)

        results = []

        for mask, box in zip(masks, boxes_640):
            mask = cv2.resize(
                mask,
                (SIZE, SIZE),
                interpolation=cv2.INTER_LINEAR,
            )

            binary = mask > 0

            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, x2 = np.clip([x1, x2], 0, SIZE)
            y1, y2 = np.clip([y1, y2], 0, SIZE)

            cropped = np.zeros(binary.shape, dtype=np.uint8)
            cropped[y1:y2, x1:x2] = binary[y1:y2, x1:x2]

            unpadded = cropped[
                top:top + valid_height,
                left:left + valid_width,
            ]

            native = cv2.resize(
                unpadded,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

            contours, _ = cv2.findContours(
                native,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            results.append([
                contour[:, 0, :].tolist()
                for contour in contours
                if len(contour) >= 3
            ])

        return results

    def segment(self, image):
        tensor, gain, left, top = self.preprocess(image)

        outputs = self.session.run(
            None,
            {self.input.name: tensor},
        )

        detections = np.asarray(outputs[0])
        prototypes = np.asarray(outputs[1])

        if detections.ndim != 3 or detections.shape[-1] != 38:
            raise RuntimeError(
                f"Expected [1,300,38], got {detections.shape}"
            )

        detections = detections[0]
        prototypes = prototypes[0]

        detections = detections[
            detections[:, 4] >= CONFIDENCE
        ]

        boxes_640 = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5].astype(np.int32)
        coefficients = detections[:, 6:38]

        boxes_native = self.restore_boxes(
            boxes_640,
            gain,
            left,
            top,
            image.shape,
        )

        polygons = self.create_polygons(
            prototypes,
            coefficients,
            boxes_640,
            image.shape,
            gain,
            left,
            top,
        )

        return [
            {
                "box_xyxy": box.astype(float).tolist(),
                "confidence": float(score),
                "class_id": int(class_id),
                "segments": segments,
            }
            for box, score, class_id, segments in zip(
                boxes_native,
                scores,
                classes,
                polygons,
            )
        ]


engine = BF16Segmenter()
inference_lock = asyncio.Lock()


def decode_message(message):
    if isinstance(message, bytes):
        encoded = message
    else:
        payload = json.loads(message)
        encoded = base64.b64decode(
            payload["image"],
            validate=True,
        )

    array = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid JPEG or PNG payload")

    return image


async def handler(websocket):
    async for message in websocket:
        request_start = time.perf_counter()

        try:
            image = decode_message(message)

            async with inference_lock:
                inference_start = time.perf_counter()

                detections = await asyncio.to_thread(
                    engine.segment,
                    image,
                )

                inference_ms = (
                    time.perf_counter() - inference_start
                ) * 1000

            response = {
                "status": "success",
                "precision": "automatic_bf16",
                "inference_time_ms": inference_ms,
                "server_time_ms": (
                    time.perf_counter() - request_start
                ) * 1000,
                "count": len(detections),
                "detections": detections,
            }

        except Exception as error:
            log.exception("Request failed")
            response = {
                "status": "error",
                "error": str(error),
            }

        await websocket.send(json.dumps(response))


async def main():
    os.makedirs("/workspace/npu_cache", exist_ok=True)
    os.makedirs("/workspace/profiles", exist_ok=True)

    async with websockets.serve(
        handler,
        "0.0.0.0",
        8765,
        max_size=10 * 1024 * 1024,
        compression=None,
    ):
        log.info("Listening on ws://0.0.0.0:8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
```

### Validation

Check the script before starting it:

```bash
python -m py_compile server.py
```

No output indicates that the script passed Python syntax validation.

**References:** [Ultralytics segmentation mask operations](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py), [websockets documentation](https://websockets.readthedocs.io/)

---

# Phase 6: Start and test the service

## Step 6.1: Start the server

Inside the container:

```bash
source /opt/ryzen-ai/venv/bin/activate
source /opt/xilinx/xrt/setup.sh

export MODEL_PATH=/workspace/yolo26n-seg.onnx
export VAIML_CONFIG=/workspace/vaiml_config.json
export ORT_PROFILE=1
export AI_ANALYZER=1

python /workspace/server.py
```

Expected startup output includes:

```text
Providers: [...]
Outputs: [...]
Listening on ws://0.0.0.0:8765
```

The first startup can take longer while the model is compiled. Later starts should load the compiled cache.

Clear the cache when the model, compiler configuration, SDK, XRT, driver or firmware changes:

```bash
rm -rf /workspace/npu_cache/*
```

### Validation

From the Fedora host:

```bash
ss -ltn | grep 8765
```

Port `8765` should be listening.

**References:** [AMD Ryzen AI compiled model caching](https://ryzenai.docs.amd.com/en/latest/modelrun.html)

---

## Step 6.2: Test from the Fedora host

Create `test_client_host.py`:

```python
import asyncio
import json
import statistics
import time

import cv2
import websockets


async def main():
    image = cv2.imread("sample.jpg")
    if image is None:
        raise RuntimeError("Unable to read sample.jpg")

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not ok:
        raise RuntimeError("JPEG encoding failed")

    samples = []

    async with websockets.connect(
        "ws://localhost:8765",
        max_size=10 * 1024 * 1024,
        compression=None,
    ) as websocket:
        for index in range(25):
            start = time.perf_counter()

            await websocket.send(encoded.tobytes())
            response = json.loads(await websocket.recv())

            rtt_ms = (
                time.perf_counter() - start
            ) * 1000

            if response["status"] != "success":
                raise RuntimeError(response)

            if index >= 5:
                samples.append(rtt_ms)

            print(
                f"{index:02d}: "
                f"inference={response['inference_time_ms']:.2f} ms, "
                f"server={response['server_time_ms']:.2f} ms, "
                f"RTT={rtt_ms:.2f} ms, "
                f"instances={response['count']}"
            )

    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]

    print(f"Warm p50 RTT: {p50:.2f} ms")
    print(f"Warm p95 RTT: {p95:.2f} ms")


asyncio.run(main())
```

Install the host dependencies and run it:

```bash
python -m pip install opencv-python-headless websockets
python test_client_host.py
```

### Validation

Confirm that:

- Every response has `status: success`.
- The response includes boxes, class IDs and segmentation contours.
- Warm p50 and p95 latency are reported.
- Returned polygons align with the source image.

**References:** [websockets client documentation](https://websockets.readthedocs.io/)

---

## Step 6.3: Observe NPU activity

While the client sends requests, run:

```bash
watch -n 0.5 \
    'xrt-smi examine --report aie-partitions'
```

### Validation

The AIE partition report should show activity while inference requests are running.

This confirms active NPU execution. Provider profiling in the next step identifies graph assignment in more detail.

**References:** [AMD XRT tools](https://xilinx.github.io/XRT/master/html/xrt_smi.html)

---

## Step 6.4: Inspect provider assignment

Locate the latest profile:

```bash
find /workspace/profiles \
    -type f \
    -name '*.json' \
    -printf '%T@ %p\n' |
    sort -nr |
    head
```

Create `inspect_profile.py`:

```python
import collections
import glob
import json
import os

profiles = glob.glob("/workspace/profiles/*.json")

if not profiles:
    raise RuntimeError("No ORT profiles found")

profile = max(profiles, key=os.path.getmtime)

with open(profile) as file:
    events = json.load(file)

providers = collections.Counter()
operations = collections.defaultdict(collections.Counter)

for event in events:
    arguments = event.get("args", {})
    provider = arguments.get("provider")
    operation = arguments.get("op_name")

    if provider:
        providers[provider] += 1

        if operation:
            operations[provider][operation] += 1

print("Profile:", profile)
print("Provider events:")

for provider, count in providers.items():
    print(f"  {provider}: {count}")

print("Operations:")

for provider, operator_counts in operations.items():
    print(provider)

    for operator, count in operator_counts.most_common():
        print(f"  {operator}: {count}")
```

Run:

```bash
python inspect_profile.py
```

### Validation

The profile must contain Vitis AI execution events:

```text
VitisAIExecutionProvider
```

Review any CPU events to determine which graph partitions remain on the CPU. A profile containing only CPU kernel execution indicates that the model was not assigned to an NPU partition.

**References:** [ONNX Runtime profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)

---

# Phase 7: Validate model accuracy

## Step 7.1: Record the FP32 baseline

Run the segmentation validation dataset through the FP32 ONNX model:

```bash
yolo segment val \
    model=yolo26n-seg.onnx \
    data=/workspace/dataset.yaml \
    imgsz=640 \
    batch=1
```

Record:

- Box mAP50
- Box mAP50–95
- Mask mAP50
- Mask mAP50–95
- Per-class metrics

**References:** [Ultralytics segmentation validation](https://docs.ultralytics.com/tasks/segment/)

---

## Step 7.2: Validate BF16 inference

Evaluate the compiled session using the same preprocessing and post-processing as `server.py`.

For each validation image, compare:

- Matched class IDs
- Box intersection over union
- Mask intersection over union
- Confidence differences
- Missing detections
- Additional detections
- Empty or malformed masks
- NaN or infinite values

### Acceptance criteria

- Output shapes match the FP32 model.
- Outputs contain no NaN or infinite values.
- Classes are not systematically changed.
- Masks remain aligned with the source images.
- Box and mask metric changes remain within the project’s accuracy tolerance.

**References:** [Ultralytics segmentation metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)

---

# Deployment acceptance checklist

## Host

- [ ] AMD NPU appears in `lspci`.
- [ ] `/dev/accel/accel0` exists.
- [ ] `amdxdna` is loaded.
- [ ] Host `xrt-smi validate` passes.
- [ ] Locked memory is unlimited.

## Container

- [ ] `/dev/accel/accel0` is visible.
- [ ] Container `xrt-smi validate` passes.
- [ ] Ryzen AI quick test prints `Test Finished`.
- [ ] `VitisAIExecutionProvider` is available.

## Model

- [ ] Input shape is `[1,3,640,640]`.
- [ ] Detection output is `[1,300,38]`.
- [ ] Prototype output is `[1,32,160,160]`.
- [ ] CPU FP32 inference succeeds.
- [ ] FP32 reference outputs are saved.

## Compilation

- [ ] VAIML session creation succeeds.
- [ ] Compiled cache files are generated.
- [ ] Cached startup succeeds.
- [ ] Outputs contain finite values.
- [ ] ORT profiling contains Vitis AI execution events.
- [ ] CPU-assigned graph partitions are understood.

## Service

- [ ] WebSocket server listens on port `8765`.
- [ ] Binary JPEG requests succeed.
- [ ] Responses contain boxes, classes and contours.
- [ ] Contours align with original image coordinates.
- [ ] Warm p50 and p95 latency are recorded.

## Accuracy

- [ ] BF16 and FP32 output shapes match.
- [ ] Representative tensor comparisons pass.
- [ ] Box metrics remain within tolerance.
- [ ] Mask metrics remain within tolerance.
- [ ] Visual segmentation checks pass.
