import os
import cv2
import streamlit as st
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(
    page_title="YOLO Object Tracker",
    page_icon="🎯"
)

st.title("🎯 Real-Time Object Tracking with YOLO")
st.markdown("Track objects in real-time using your webcam stream with Ultralytics YOLO.")

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

# --- Load YOLO Model ---
MODEL_PATH = "puck-eye-s.pt"

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = load_model(MODEL_PATH)
if model is None:
    st.error(f"Model file `{MODEL_PATH}` not found in root directory.")

# --- Webcam Selection ---
available_cams = get_available_cameras()
selected_cam = st.selectbox(
    "Select Webcam",
    options=available_cams,
    format_func=lambda x: f"Webcam Device {x}"
)

# --- Main Inference Loop (Single Column Layout) ---
st.subheader("Live Feed")
frame_placeholder = st.empty()

st.subheader("Tracking Metrics")
fps_metric = st.empty()
count_metric = st.empty()

start_button = st.button("🚀 Start Tracking", type="primary", use_container_width=True)
stop_button = st.button("⏹️ Stop Tracking", use_container_width=True)

if start_button and model is not None:
    # Run tracking stream generator
    results_generator = model.track(
        source=selected_cam,
        tracker="bytetrack.yaml",
        conf=0.4,
        iou=0.5,
        imgsz=640,
        stream=True,     # Stream results frame-by-frame
        verbose=False
    )

    prev_time = cv2.getTickCount()

    for result in results_generator:
        if stop_button:
            break

        # Annotated image as RGB numpy array
        annotated_frame = result.plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Calculate FPS
        curr_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_time - prev_time)
        prev_time = curr_time

        # Get track IDs / count active tracked objects
        tracked_ids = result.boxes.id
        num_objects = len(tracked_ids) if tracked_ids is not None else 0

        # Update Streamlit UI
        frame_placeholder.image(annotated_frame_rgb, channels="RGB", width="stretch")
        fps_metric.metric("FPS", f"{fps:.1f}")
        count_metric.metric("Active Tracked Objects", f"{num_objects}")

    st.info("Tracking session ended.")