# YOLO26 Real-Time Object Detection

A Streamlit web application for real-time object detection and segmentation using YOLOE-26L with text prompts. Detect any object by simply typing what you're looking for.

## Features

- **Text-Prompted Detection**: Enter any object class (e.g., "person", "car", "dog") and the model will detect it
- **Multiple Input Modes**:
  - Upload Image: Process static images
  - Upload Video: Process video files with configurable frame skip
  - Live Camera: Real-time detection via webcam
- **Instance Segmentation**: Uses YOLOE-26L-Seg model for pixel-level object segmentation

## Requirements

- Python 3.10+
- Webcam (for live camera mode)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd realtime-detection-yolo26
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure the model file `yoloe-26l-seg.pt` is in the project directory.

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Modes

**Upload Image**
1. Select "Upload Image" from the dropdown
2. Upload a JPG, PNG, or JPEG image
3. Enter the object class to detect
4. View the detection results with segmentation masks

**Upload Video**
1. Select "Upload Video" from the dropdown
2. Upload an MP4, AVI, or MOV video file
3. Enter the object class to detect
4. Adjust the frame skip slider (higher values = faster processing)
5. Watch the processed video with detections

**Live Camera**
1. Select "Live Camera" from the dropdown
2. Enter the object class to detect
3. Allow browser access to your webcam
4. View real-time detections

## Model

This application uses **YOLOE-26L-Seg**, a text-promptable object detection and segmentation model from Ultralytics. The model supports open-vocabulary detection, allowing you to detect objects by their text descriptions rather than being limited to predefined classes.

## Project Structure

```
realtime-detection-yolo26/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── yoloe-26l-seg.pt    # YOLO model weights
├── mobileclip2_b.ts    # MobileCLIP text encoder
└── README.md           # This file
```

## License

See the [Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for model usage terms.
