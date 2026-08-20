FROM ryzerdocker:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_PRELOAD=/opt/xilinx/xrt/lib/libxrt_coreutil.so.2
WORKDIR /workspace

# Copy model assets
COPY docker-data/yolo26s-seg.pt /workspace/yolo26s-seg.pt
COPY docker-data/sample.jpg /workspace/sample.jpg

# Copy Ryzen AI SDK archive
COPY docker-data/ryzen_ai-1.8.0.tgz /tmp/

# Extract and install AMD Ryzen AI SDK, and preserve the onnxruntime_vitisai wheel
RUN mkdir -p /tmp/ryzen-ai /opt/ryzen-ai && \
    tar -xzf /tmp/ryzen_ai-1.8.0.tgz -C /tmp/ryzen-ai && \
    cp /tmp/ryzen-ai/onnxruntime_vitisai-1.27.0-py3-none-linux_x86_64.whl /tmp/onnxruntime_vitisai-1.27.0-py3-none-linux_x86_64.whl && \
    cd /tmp/ryzen-ai && \
    ./install_ryzen_ai.sh -a yes -p /opt/ryzen-ai/venv && \
    rm -rf /tmp/ryzen-ai

# Activate venv and install required Python packages
RUN . /opt/ryzen-ai/venv/bin/activate && \
    pip install --no-cache-dir \
        "numpy<2" \
        "opencv-python-headless<4.11" \
        "opencv-python<4.11" \
        ultralytics \
        websockets \
        onnx \
        onnxslim

# Ensure AMD's custom onnxruntime-vitisai is the active onnxruntime package
RUN . /opt/ryzen-ai/venv/bin/activate && \
    pip uninstall -y onnxruntime onnxruntime-vitisai && \
    pip install --no-deps /tmp/onnxruntime_vitisai-1.27.0-py3-none-linux_x86_64.whl && \
    rm -f /tmp/onnxruntime_vitisai-1.27.0-py3-none-linux_x86_64.whl /tmp/ryzen_ai-1.8.0.tgz

# Copy server application files
COPY npu/vaiml_config.json /workspace/vaiml_config.json
COPY npu/export_model.py /workspace/export_model.py
COPY npu/server.py /workspace/server.py
COPY npu/entrypoint.sh /workspace/entrypoint.sh

RUN chmod +x /workspace/entrypoint.sh

# Bake static FP32 ONNX and QDQ INT8 models during build time
RUN . /opt/ryzen-ai/venv/bin/activate && \
    export PT_MODEL_PATH=/workspace/yolo26s-seg.pt && \
    export ONNX_MODEL_PATH=/workspace/yolo26s-seg.onnx && \
    export QDQ_MODEL_PATH=/workspace/yolo26s-seg_qdq.onnx && \
    python /workspace/export_model.py

EXPOSE 8765

ENTRYPOINT ["/workspace/entrypoint.sh"]
