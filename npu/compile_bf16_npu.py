import os
import onnx
import onnxruntime as ort
from ultralytics import YOLO

# 1. Export Standard FP32 Static ONNX
model = YOLO("puck-eye-seg-s.pt")
onnx_fp32_path = "puck-eye-seg-s_fp32.onnx"
model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    simplify=True,
    opset=17
)

# 2. Create VAI EP configuration for BF16 Native compilation
vaip_config_content = """{
    "target": "X2",
    "compiler": {
        "opt_level": 3,
        "precision": "BF16"
    }
}"""

with open("vaip_bf16_config.json", "w") as f:
    f.write(vaip_config_content)

# 3. Compile and Export EP-Context Model for XDNA 2 NPU
print("[INFO] Compiling graph down to Native XDNA 2 BF16 instructions...")
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Direct ORT to dump the precompiled EP Context model
compiled_output_path = "puck-eye-seg-s_bf16_ctx.onnx"
session_options.add_session_config_entry("ep.context_enable", "1")
session_options.add_session_config_entry("ep.context_file_path", compiled_output_path)
session_options.add_session_config_entry("ep.context_embed_mode", "1")

provider_options = [{
    "config_file": os.path.abspath("vaip_bf16_config.json"),
    "cacheDir": os.path.abspath("./npu_cache_bf16")
}]

# Instantiating the session triggers offline compilation and writes the context file
session = ort.InferenceSession(
    onnx_fp32_path,
    sess_options=session_options,
    providers=["VitisAIExecutionProvider"],
    provider_options=provider_options
)

print(f"[SUCCESS] Precompiled BF16 NPU model created: {compiled_output_path}")