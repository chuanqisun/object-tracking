# YOLO26 Real-Time Object Detection

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
streamlit run app.py # stock yoloe text prompt tracking

streamlit run app-custom.py # custom trained model for speaker tracking
```

## Extra

Tradition CSRT tracker are in `web` directory for comparison.
