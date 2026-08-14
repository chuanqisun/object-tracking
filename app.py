import streamlit as st
import streamlit.elements.image as st_image
from dataclasses import dataclass

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

from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import numpy as np
import os
import tempfile
import av
from streamlit_webrtc import webrtc_streamer

@st.cache_resource
def load_model():
    model = YOLO("yoloe-26s-seg.pt")
    return model

model = load_model()

class VideoProcessor:
    prompt_mode = "Text Prompt"
    current_class = ''
    ref_img = None
    visual_prompts = None

    @classmethod
    def update_text_prompt(cls, new_class):
        cls.prompt_mode = "Text Prompt"
        cls.current_class = new_class
        names = [new_class]
        model.set_classes(names, model.get_text_pe(names))

    @classmethod
    def update_visual_prompt(cls, ref_img, bboxes, text_label=''):
        cls.prompt_mode = "Visual Prompt"
        cls.ref_img = ref_img
        cls.visual_prompts = {'bboxes': bboxes, 'cls': [0]}
        cls.current_class = text_label.strip()
        if cls.current_class:
            names = [cls.current_class]
            model.set_classes(names, model.get_text_pe(names))

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if VideoProcessor.prompt_mode == "Text Prompt" and VideoProcessor.current_class:
            results = model.predict(img)
            plotted_img = results[0].plot()
        elif (
            VideoProcessor.prompt_mode == "Visual Prompt"
            and VideoProcessor.ref_img is not None
            and VideoProcessor.visual_prompts is not None
        ):
            results = model.predict(
                img,
                refer_image=VideoProcessor.ref_img,
                visual_prompts=VideoProcessor.visual_prompts,
                predictor=YOLOEVPSegPredictor,
            )
            plotted_img = results[0].plot()
        else:
            plotted_img = img  # no detection
        return av.VideoFrame.from_ndarray(plotted_img, format="bgr24")


def extract_bboxes_from_canvas(canvas_result, orig_w, orig_h, canvas_w, canvas_h):
    if canvas_result is None or canvas_result.json_data is None:
        return []
    scale_x = orig_w / canvas_w
    scale_y = orig_h / canvas_h
    bboxes = []
    for obj in canvas_result.json_data.get("objects", []):
        if obj.get("type") == "rect":
            left = obj["left"]
            top = obj["top"]
            width = obj["width"] * obj.get("scaleX", 1)
            height = obj["height"] * obj.get("scaleY", 1)
            x1 = max(0, left * scale_x)
            y1 = max(0, top * scale_y)
            x2 = min(orig_w, (left + width) * scale_x)
            y2 = min(orig_h, (top + height) * scale_y)
            if x2 > x1 + 5 and y2 > y1 + 5:
                bboxes.append([x1, y1, x2, y2])
    return bboxes


def setup_canvas_drawing(pil_image, key_suffix=""):
    st.markdown("#### ✏️ Draw a rectangle around the object to use as visual prompt")
    orig_w, orig_h = pil_image.size
    max_canvas_width = 700
    canvas_w = min(max_canvas_width, orig_w)
    canvas_h = int(orig_h * (canvas_w / orig_w))

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=pil_image,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="rect",
        key=f"canvas_{key_suffix}",
    )

    bboxes = extract_bboxes_from_canvas(canvas_result, orig_w, orig_h, canvas_w, canvas_h)
    if bboxes:
        bbox = bboxes[0]
        crop_img = np.array(pil_image)[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(crop_img, caption="Cropped Prompt Object", width=150)
        with col2:
            st.success(f"Selected Prompt Box: [x1: {int(bbox[0])}, y1: {int(bbox[1])}, x2: {int(bbox[2])}, y2: {int(bbox[3])}]")
    else:
        st.info("💡 Draw a box over the object on the canvas above.")

    return bboxes


if 'running' not in st.session_state:
    st.session_state.running = False

st.title("YOLO26 Object Detection")

col_mode, col_prompt = st.columns([1, 1])
with col_mode:
    mode = st.selectbox("Select mode", ["Upload Image", "Upload Video", "Live Camera"])
with col_prompt:
    prompt_type = st.radio("Select Prompt Type", ["Text Prompt", "Visual Prompt (Draw Box)"], horizontal=True)


if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        if prompt_type == "Text Prompt":
            word = st.text_input("Enter a class to detect (e.g., person, bus)")
            if word.strip():
                names = [word.strip()]
                model.set_classes(names, model.get_text_pe(names))
                results = model.predict(image_bgr)
                plotted_img = results[0].plot()
                st.image(plotted_img, channels="BGR", use_container_width=True)
        else:
            word = st.text_input("Optional: Enter object name/label for the drawn visual prompt (e.g., mug)")
            bboxes = setup_canvas_drawing(pil_image, key_suffix="img")
            if bboxes and st.button("Detect Similar Objects"):
                if word.strip():
                    names = [word.strip()]
                    model.set_classes(names, model.get_text_pe(names))
                visual_prompts = {'bboxes': bboxes, 'cls': [0]}
                results = model.predict(
                    image_bgr,
                    refer_image=image_bgr,
                    visual_prompts=visual_prompts,
                    predictor=YOLOEVPSegPredictor,
                )
                plotted_img = results[0].plot()
                st.image(plotted_img, channels="BGR", use_container_width=True)

elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    frame_skip = st.slider("Process every Nth frame (higher = faster, less detailed)", 1, 10, 5)

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open video file")
        else:
            if prompt_type == "Text Prompt":
                word = st.text_input("Enter a class to detect (e.g., person, bus)")
                if word.strip() and st.button("Start Processing Video"):
                    names = [word.strip()]
                    model.set_classes(names, model.get_text_pe(names))
                    frame_placeholder = st.empty()
                    st.write("Processing video...")
                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_count += 1
                        if frame_count % frame_skip != 0:
                            continue
                        results = model.predict(frame)
                        plotted_img = results[0].plot()
                        frame_placeholder.image(plotted_img, channels="BGR", use_container_width=True)
                    cap.release()
                    st.write("Video processing completed.")
            else:
                word = st.text_input("Optional: Enter object name/label for the drawn visual prompt (e.g., mug)")
                ret, first_frame = cap.read()
                cap.release()
                if ret:
                    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                    pil_ref = Image.fromarray(first_frame_rgb)
                    bboxes = setup_canvas_drawing(pil_ref, key_suffix="vid")

                    if bboxes and st.button("Start Processing Video with Visual Prompt"):
                        if word.strip():
                            names = [word.strip()]
                            model.set_classes(names, model.get_text_pe(names))
                        cap_proc = cv2.VideoCapture(video_path)
                        frame_placeholder = st.empty()
                        st.write("Processing video using Visual Prompt...")
                        frame_count = 0
                        visual_prompts = {'bboxes': bboxes, 'cls': [0]}
                        while True:
                            ret_p, frame_p = cap_proc.read()
                            if not ret_p:
                                break
                            frame_count += 1
                            if frame_count % frame_skip != 0:
                                continue
                            results = model.predict(
                                frame_p,
                                refer_image=first_frame,
                                visual_prompts=visual_prompts,
                                predictor=YOLOEVPSegPredictor,
                            )
                            plotted_img = results[0].plot()
                            frame_placeholder.image(plotted_img, channels="BGR", use_container_width=True)
                        cap_proc.release()
                        st.write("Video processing completed.")

elif mode == "Live Camera":
    if prompt_type == "Text Prompt":
        word = st.text_input("Enter a class to detect (e.g., person, bus)")
        if word.strip():
            VideoProcessor.update_text_prompt(word.strip())
            webrtc_streamer(
                key="yolo-live-text",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
            )
    else:
        word = st.text_input("Optional: Enter object name/label for the drawn visual prompt (e.g., mug)")
        st.write("📸 **Step 1: Capture or Upload Reference Snapshot**")
        snapshot = st.camera_input("Take a photo to select an object prompt")
        uploaded_snap = st.file_uploader("Or upload a reference photo", type=["jpg", "png", "jpeg"], key="ref_snap")

        ref_pil = None
        if snapshot is not None:
            ref_pil = Image.open(snapshot).convert("RGB")
        elif uploaded_snap is not None:
            ref_pil = Image.open(uploaded_snap).convert("RGB")

        if ref_pil is not None:
            st.write("🎯 **Step 2: Draw a bounding box around your object of interest**")
            bboxes = setup_canvas_drawing(ref_pil, key_suffix="cam")
            if bboxes:
                st.write("📹 **Step 3: Start Live Camera Detection**")
                ref_bgr = cv2.cvtColor(np.array(ref_pil), cv2.COLOR_RGB2BGR)
                VideoProcessor.update_visual_prompt(ref_bgr, bboxes, text_label=word)
                webrtc_streamer(
                    key="yolo-live-visual",
                    video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                )
