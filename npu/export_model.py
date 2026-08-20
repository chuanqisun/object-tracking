import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.quantization as ort_quant
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import ultralytics
import ultralytics.utils.checks

# Prevent ultralytics from auto-installing stock onnxruntime
ultralytics.utils.checks.check_requirements = lambda *args, **kwargs: True

try:
    ultralytics.settings.update({"auto_update": False})
except Exception as e:
    print(f"Warning: Could not update ultralytics settings: {e}")

from ultralytics import YOLO

class DummyCalibrationDataReader(CalibrationDataReader):
    """Dummy calibration data reader that generates synthetic image tensors (no real dataset needed)."""
    def __init__(self, num_samples=3, shape=(1, 3, 640, 640)):
        self.data = [{"images": np.random.randn(*shape).astype(np.float32)} for _ in range(num_samples)]
        self.enum_data = iter(self.data)

    def get_next(self):
        return next(self.enum_data, None)

def export():
    pt_path = os.getenv("PT_MODEL_PATH", "/workspace/yolo26s-seg.pt")
    onnx_path = os.getenv("ONNX_MODEL_PATH", "/workspace/yolo26s-seg.onnx")
    qdq_path = os.getenv("QDQ_MODEL_PATH", "/workspace/yolo26s-seg_qdq.onnx")

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PyTorch model file not found at {pt_path}")

    print(f"Loading model from {pt_path}...")
    model = YOLO(pt_path)

    print("Exporting model to static FP32 ONNX...")
    exported_file = model.export(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        simplify=True,
        opset=17,
        end2end=True,
        half=False,
    )

    print(f"Exported FP32 ONNX model to: {exported_file}")

    # Verify FP32 ONNX model
    graph = onnx.load(exported_file)
    onnx.checker.check_model(graph)

    # Perform QDQ INT8 Quantization (required for AMD NPU hardware acceleration)
    print(f"Quantizing FP32 ONNX to QDQ INT8 format ({qdq_path}) using synthetic calibration data...")
    quantize_static(
        exported_file,
        qdq_path,
        DummyCalibrationDataReader(num_samples=3, shape=(1, 3, 640, 640)),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    print(f"Exported QDQ INT8 model to: {qdq_path}")

    qdq_graph = onnx.load(qdq_path)
    onnx.checker.check_model(qdq_graph)
    print(f"QDQ INT8 model successfully verified ({len(qdq_graph.graph.node)} nodes).")

if __name__ == "__main__":
    export()
