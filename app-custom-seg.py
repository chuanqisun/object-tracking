import os
import tempfile
from dataclasses import dataclass

import av
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.elements.image as st_image
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

# --- Patch Streamlit image_to_url for version compatibility ---
@dataclass
class LayoutConfig:
    width: int = None

try:
    import streamlit.elements.lib.image_utils as iu
    def patched_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="PNG", image_id=""):
        config = LayoutConfig(width=width if isinstance(width, int) else None)
        return iu.image_to_url(
            image=image,
            layout_config=config,
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )
    st_image.image_to_url = patched_image_to_url
except Exception:
    pass

# --- Page Configuration ---
st.set_page_config(
    page_title="YOLO26 Instance Segmentation",
    page_icon="🧩",
    layout="wide"
)

st.title("🧩 YOLO26 Real-Time Instance Segmentation & Tracking")
st.markdown(
    "Alternative segmentation application powered by **`yolo26s-seg.pt`** (Ultralytics YOLO26 Instance Segmentation). "
    "Supports image analysis, video tracking, local webcam, and WebRTC streaming."
)

# --- Load YOLO26 Segment Model ---
MODEL_PATH = "puck-eye-seg-s.pt"

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.info(f"Model file `{path}` not found locally. Loading/downloading standard YOLO model...")
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model `{MODEL_PATH}`: {e}")
    st.stop()

# --- Helper to list available webcams ---
@st.cache_data
def get_available_cameras(max_tested=5):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available if available else [0]

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Model & Segmentation Options")

# Mode Selection
mode = st.sidebar.selectbox(
    "Select Operating Mode",
    ["Upload Image", "Upload Video", "Live Webcam (Direct)", "Live WebRTC Stream"]
)

st.sidebar.subheader("Confidence & IoU Thresholds")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
iou_thresh = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)

# Class Filter
all_class_names = list(model.names.values())
selected_classes = st.sidebar.multiselect(
    "Filter Classes (Leave empty to detect all classes)",
    options=all_class_names,
    default=[]
)

class_indices = None
if selected_classes:
    class_indices = [k for k, v in model.names.items() if v in selected_classes]

st.sidebar.subheader("Visualization Settings")
show_masks = st.sidebar.checkbox("Show Instance Masks", value=True)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
show_conf = st.sidebar.checkbox("Show Confidence Scores", value=True)

tracker_type = st.sidebar.selectbox("Tracker Algorithm", ["bytetrack.yaml", "botsort.yaml"])

# --- App Logic ---

if mode == "Upload Image":
    st.header("🖼️ Image Instance Segmentation")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with st.spinner("Running YOLO26 Instance Segmentation..."):
            results = model.predict(
                source=img_bgr,
                conf=conf_thresh,
                iou=iou_thresh,
                classes=class_indices,
                verbose=False
            )
            res = results[0]
            plotted_bgr = res.plot(
                masks=show_masks,
                boxes=show_boxes,
                labels=show_labels,
                conf=show_conf
            )
            plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(pil_img, use_container_width=True)
        with col2:
            st.subheader("Segmented Output")
            st.image(plotted_rgb, use_container_width=True)

        # Details & Breakdown
        num_instances = len(res.boxes) if res.boxes is not None else 0
        st.subheader(f"📊 Detected Instances: {num_instances}")

        if num_instances > 0:
            data = []
            masks_data = res.masks.data if res.masks is not None else None
            for idx, box in enumerate(res.boxes):
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"Class {cls_id}")
                conf_score = float(box.conf[0].item())
                xyxy = [round(x, 1) for x in box.xyxy[0].tolist()]

                mask_pixels = 0
                if masks_data is not None and idx < len(masks_data):
                    mask_pixels = int(masks_data[idx].sum().item())

                data.append({
                    "Instance": idx + 1,
                    "Class": cls_name,
                    "Confidence": f"{conf_score:.2f}",
                    "Bounding Box [x1, y1, x2, y2]": str(xyxy),
                    "Mask Area (Pixels)": mask_pixels
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.info("No objects detected with current thresholds.")

elif mode == "Upload Video":
    st.header("🎥 Video Instance Segmentation & Tracking")
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    col_v1, col_v2 = st.columns(2)
    frame_skip = col_v1.slider("Process every Nth frame (higher = faster)", 1, 10, 2)
    use_tracking = col_v2.checkbox("Enable Object Tracking (ByteTrack / BoT-SORT)", value=True)

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open video file.")
        else:
            col_b1, col_b2 = st.columns([1, 4])
            start_btn = col_b1.button("🎬 Start Processing Video", type="primary")

            if start_btn:
                st.write("Processing video frames...")
                frame_placeholder = st.empty()
                m1, m2 = st.columns(2)
                fps_metric = m1.empty()
                count_metric = m2.empty()

                frame_count = 0
                prev_time = cv2.getTickCount()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue

                    if use_tracking:
                        results = model.track(
                            source=frame,
                            conf=conf_thresh,
                            iou=iou_thresh,
                            classes=class_indices,
                            tracker=tracker_type,
                            persist=True,
                            verbose=False
                        )
                    else:
                        results = model.predict(
                            source=frame,
                            conf=conf_thresh,
                            iou=iou_thresh,
                            classes=class_indices,
                            verbose=False
                        )

                    res = results[0]
                    plotted = res.plot(
                        masks=show_masks,
                        boxes=show_boxes,
                        labels=show_labels,
                        conf=show_conf
                    )
                    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

                    curr_time = cv2.getTickCount()
                    fps = cv2.getTickFrequency() / max(curr_time - prev_time, 1)
                    prev_time = curr_time

                    num_objects = len(res.boxes) if res.boxes is not None else 0

                    frame_placeholder.image(plotted_rgb, channels="RGB", use_container_width=True)
                    fps_metric.metric("FPS", f"{fps:.1f}")
                    count_metric.metric("Active Detected Instances", f"{num_objects}")

                cap.release()
                st.success("Video processing completed.")

elif mode == "Live Webcam (Direct)":
    st.header("📹 Direct Webcam Tracking")
    available_cams = get_available_cameras()
    selected_cam = st.selectbox(
        "Select Webcam Device",
        options=available_cams,
        format_func=lambda x: f"Webcam Device {x}"
    )

    use_tracking = st.checkbox("Enable Track Persistence", value=True)

    frame_placeholder = st.empty()
    col_f, col_c = st.columns(2)
    fps_metric = col_f.empty()
    count_metric = col_c.empty()

    col_s1, col_s2 = st.columns(2)
    start_tracking = col_s1.button("🚀 Start Tracking", type="primary", use_container_width=True)
    stop_tracking = col_s2.button("⏹️ Stop Tracking", use_container_width=True)

    if start_tracking:
        cap = cv2.VideoCapture(selected_cam)
        if not cap.isOpened():
            st.error(f"Could not open camera device {selected_cam}.")
        else:
            prev_time = cv2.getTickCount()
            while cap.isOpened():
                if stop_tracking:
                    break
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame from camera.")
                    break

                if use_tracking:
                    results = model.track(
                        source=frame,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        classes=class_indices,
                        tracker=tracker_type,
                        persist=True,
                        verbose=False
                    )
                else:
                    results = model.predict(
                        source=frame,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        classes=class_indices,
                        verbose=False
                    )

                res = results[0]
                plotted = res.plot(
                    masks=show_masks,
                    boxes=show_boxes,
                    labels=show_labels,
                    conf=show_conf
                )
                plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

                curr_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / max(curr_time - prev_time, 1)
                prev_time = curr_time

                num_instances = len(res.boxes) if res.boxes is not None else 0

                frame_placeholder.image(plotted_rgb, channels="RGB", use_container_width=True)
                fps_metric.metric("FPS", f"{fps:.1f}")
                count_metric.metric("Active Instances", f"{num_instances}")

            cap.release()
            st.info("Live tracking stopped.")

elif mode == "Live WebRTC Stream":
    st.header("🌐 WebRTC Browser Camera Streaming")
    st.markdown("Stream video directly from your browser with real-time segmentation.")

    class SegmentationVideoProcessor:
        def __init__(self):
            self.conf = conf_thresh
            self.iou = iou_thresh
            self.classes = class_indices
            self.show_masks = show_masks
            self.show_boxes = show_boxes
            self.show_labels = show_labels
            self.show_conf = show_conf

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(
                source=img,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                verbose=False
            )
            plotted_img = results[0].plot(
                masks=self.show_masks,
                boxes=self.show_boxes,
                labels=self.show_labels,
                conf=self.show_conf
            )
            return av.VideoFrame.from_ndarray(plotted_img, format="bgr24")

    webrtc_streamer(
        key="yolo26-seg-webrtc",
        video_processor_factory=SegmentationVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
