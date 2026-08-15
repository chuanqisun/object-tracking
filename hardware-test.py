import torch
import logging

# --- Check Hardware Acceleration ---
logging.info("🖥️ Hardware Info")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logging.info(f"Running on **GPU**: {gpu_name} ({vram_gb:.1f} GB)")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    logging.info("Running on **Apple Silicon (MPS)**")
else:
    logging.warning("Running on **CPU** (No GPU detected)")

