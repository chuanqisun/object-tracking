# syntax=docker/dockerfile:1
FROM ryzerdocker:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_PRELOAD=/opt/xilinx/xrt/lib/libxrt_coreutil.so.2
# Forces real-time output for python logs during build and runtime
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

# Copy model assets
COPY docker-data/yolo26s-seg.pt /workspace/yolo26s-seg.pt
COPY docker-data/sample.jpg /workspace/sample.jpg

# Extract and install AMD Ryzen AI SDK, Python dependencies, and clean build caches in the same layer
RUN --mount=type=bind,source=docker-data/ryzen_ai-1.8.0.tgz,target=/tmp/ryzen_ai-1.8.0.tgz \
    mkdir -p /tmp/ryzen-ai /opt/ryzen-ai && \
    tar -xzf /tmp/ryzen_ai-1.8.0.tgz -C /tmp/ryzen-ai && \
    cp /tmp/ryzen-ai/onnxruntime_vitisai-1.27.0-py3-none-linux_x86_64.whl /tmp/ && \
    cd /tmp/ryzen-ai && \
    ./install_ryzen_ai.sh -a yes -p /opt/ryzen-ai/venv && \
    cd /workspace && \
    rm -rf /tmp/ryzen-ai && \
    . /opt/ryzen-ai/venv/bin/activate && \
    pip install --no-cache-dir --progress-bar on \
        "numpy<2" \
        "opencv-python-headless<4.11" \
        ultralytics \
        websockets \
        onnx \
        onnxslim && \
    pip uninstall -y onnxruntime onnxruntime-vitisai && \
    pip install --no-deps /tmp/onnxruntime_vitisai-1.27.0-py3-none-linux_x86_64.whl && \
    rm -f /tmp/onnxruntime_vitisai-1.27.0-py3-none-linux_x86_64.whl && \
    # Strip unnecessary bytecode/tests to dramatically shrink layer size
    find /opt/ryzen-ai/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/ryzen-ai/venv -type f -name "*.pyc" -delete && \
    find /opt/ryzen-ai/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true

# Copy server application files
COPY npu/vaiml_config.json /workspace/vaiml_config.json
COPY npu/export_model.py /workspace/export_model.py
COPY npu/server.py /workspace/server.py
COPY npu/entrypoint.sh /workspace/entrypoint.sh

RUN chmod +x /workspace/entrypoint.sh

# Bake static FP32 ONNX and QDQ INT8 models with real-time logs and post-cleanup
RUN . /opt/ryzen-ai/venv/bin/activate && \
    export PT_MODEL_PATH=/workspace/yolo26s-seg.pt && \
    export ONNX_MODEL_PATH=/workspace/yolo26s-seg.onnx && \
    export QDQ_MODEL_PATH=/workspace/yolo26s-seg_qdq.onnx && \
    python -u /workspace/export_model.py && \
    find /workspace -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /workspace -type f -name "*.pyc" -delete

EXPOSE 8765

ENTRYPOINT ["/workspace/entrypoint.sh"]
