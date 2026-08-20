import asyncio
import base64
import json
import logging
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("yolo26-npu-server")

SIZE = 640
CONF_DEFAULT = 0.35


def letterbox(image):
    height, width = image.shape[:2]
    gain = min(SIZE / height, SIZE / width)

    new_width = round(width * gain)
    new_height = round(height * gain)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)

    left = (SIZE - new_width) // 2
    top = (SIZE - new_height) // 2

    canvas[
        top:top + new_height,
        left:left + new_width,
    ] = resized

    return canvas, gain, left, top


class BF16Segmenter:
    def __init__(self):
        default_model = (
            "/workspace/yolo26s-seg_qdq.onnx"
            if os.path.exists("/workspace/yolo26s-seg_qdq.onnx")
            else "/workspace/yolo26s-seg.onnx"
        )
        default_config = (
            "/opt/ryzen-ai/venv/lib/python3.12/site-packages/voe/bin/vaip_config.json"
            if os.path.exists("/opt/ryzen-ai/venv/lib/python3.12/site-packages/voe/bin/vaip_config.json")
            else "/workspace/vaiml_config.json"
        )

        model = os.getenv("MODEL_PATH", default_model)
        config = os.getenv("VAIML_CONFIG", default_config)
        cache_dir = os.getenv("NPU_CACHE_DIR", "/workspace/npu_cache")
        os.makedirs(cache_dir, exist_ok=True)

        session_options = ort.SessionOptions()
        session_options.log_severity_level = int(os.getenv("ORT_LOG_LEVEL", "2"))

        if os.getenv("ORT_PROFILE", "0") == "1":
            session_options.enable_profiling = True
            session_options.profile_file_prefix = "/workspace/profiles/yolo26_npu"

        vai_options = {
            "config_file": os.path.abspath(config),
            "cacheDir": os.path.abspath(cache_dir),
            "cacheKey": "yolo26_seg_qdq_cache",
        }

        if os.getenv("AI_ANALYZER", "0") == "1":
            vai_options.update({
                "ai_analyzer_visualization": "true",
                "ai_analyzer_profiling": "true",
            })

        log.info("Initializing InferenceSession with VitisAIExecutionProvider...")
        log.info("Using model: %s", model)
        log.info("Using VAIP config: %s", config)

        self.session = ort.InferenceSession(
            model,
            sess_options=session_options,
            providers=[
                "VitisAIExecutionProvider",
                "CPUExecutionProvider",
            ],
            provider_options=[
                vai_options,
                {},
            ],
        )

        self.input = self.session.get_inputs()[0]
        self.outputs = self.session.get_outputs()

        log.info("Available Providers: %s", self.session.get_providers())
        log.info("Input: %s, shape=%s", self.input.name, self.input.shape)
        log.info("Outputs: %s", [(item.name, item.shape) for item in self.outputs])

        self.warmup()

    def warmup(self):
        log.info("Warming up model execution...")
        dummy = np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)
        for i in range(3):
            t0 = time.perf_counter()
            self.session.run(None, {self.input.name: dummy})
            log.info("Warm-up run %d: %.2f ms", i + 1, (time.perf_counter() - t0) * 1000)

    def preprocess(self, image):
        padded, gain, left, top = letterbox(image)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = (rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]
        return tensor, gain, left, top

    @staticmethod
    def restore_boxes(boxes, gain, left, top, shape):
        boxes = boxes.copy()
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes /= gain

        height, width = shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        return boxes

    @staticmethod
    def create_polygons(prototypes, coefficients, boxes_640, original_shape, gain, left, top):
        if len(coefficients) == 0:
            return []

        channels, mask_height, mask_width = prototypes.shape
        logits = coefficients @ prototypes.reshape(channels, -1)
        masks = logits.reshape(-1, mask_height, mask_width)

        original_height, original_width = original_shape[:2]
        valid_height = round(original_height * gain)
        valid_width = round(original_width * gain)

        results = []
        for mask, box in zip(masks, boxes_640):
            mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            binary = mask > 0

            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, x2 = np.clip([x1, x2], 0, SIZE)
            y1, y2 = np.clip([y1, y2], 0, SIZE)

            cropped = np.zeros(binary.shape, dtype=np.uint8)
            cropped[y1:y2, x1:x2] = binary[y1:y2, x1:x2]

            unpadded = cropped[top:top + valid_height, left:left + valid_width]

            native = cv2.resize(
                unpadded,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

            contours, _ = cv2.findContours(
                native,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            results.append([
                contour[:, 0, :].tolist()
                for contour in contours
                if len(contour) >= 3
            ])

        return results

    def segment(self, image, conf_threshold=CONF_DEFAULT):
        tensor, gain, left, top = self.preprocess(image)

        outputs = self.session.run(None, {self.input.name: tensor})

        detections = np.asarray(outputs[0])
        prototypes = np.asarray(outputs[1])

        if detections.ndim == 3:
            detections = detections[0]
        if prototypes.ndim == 4:
            prototypes = prototypes[0]

        detections = detections[detections[:, 4] >= conf_threshold]

        if len(detections) == 0:
            return []

        boxes_640 = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5].astype(np.int32)
        coefficients = detections[:, 6:38]

        boxes_native = self.restore_boxes(
            boxes_640, gain, left, top, image.shape
        )

        polygons = self.create_polygons(
            prototypes, coefficients, boxes_640, image.shape, gain, left, top
        )

        return [
            {
                "box_xyxy": box.astype(float).tolist(),
                "confidence": float(score),
                "class_id": int(class_id),
                "segments": segments,
            }
            for box, score, class_id, segments in zip(
                boxes_native, scores, classes, polygons
            )
        ]


engine = None
inference_lock = asyncio.Lock()


def decode_message(message):
    if isinstance(message, bytes):
        encoded = message
        conf_thresh = CONF_DEFAULT
    else:
        payload = json.loads(message)
        if "image" in payload:
            encoded = base64.b64decode(payload["image"], validate=True)
        else:
            raise ValueError("JSON payload missing 'image' field")
        conf_thresh = float(payload.get("confidence", CONF_DEFAULT))

    array = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image payload: unable to decode JPEG/PNG")

    return image, conf_thresh


async def handler(websocket):
    client = websocket.remote_address
    log.info("Client connected: %s", client)

    try:
        async for message in websocket:
            request_start = time.perf_counter()

            try:
                image, conf_thresh = decode_message(message)

                async with inference_lock:
                    inference_start = time.perf_counter()
                    detections = await asyncio.to_thread(
                        engine.segment, image, conf_thresh
                    )
                    inference_ms = (time.perf_counter() - inference_start) * 1000

                response = {
                    "status": "success",
                    "precision": "int8_qdq_npu",
                    "inference_time_ms": round(inference_ms, 2),
                    "server_time_ms": round((time.perf_counter() - request_start) * 1000, 2),
                    "count": len(detections),
                    "detections": detections,
                }

            except Exception as error:
                log.exception("Inference processing error")
                response = {
                    "status": "error",
                    "error": str(error),
                }

            await websocket.send(json.dumps(response))
    except websockets.ConnectionClosed:
        log.info("Client disconnected: %s", client)


async def main():
    global engine
    engine = BF16Segmenter()

    port = int(os.getenv("PORT", "8765"))
    log.info("Starting WebSocket server on 0.0.0.0:%d", port)

    async with websockets.serve(
        handler,
        "0.0.0.0",
        port,
        max_size=10 * 1024 * 1024,
        compression=None,
    ):
        log.info("Server is up and ready for connections on ws://0.0.0.0:%d", port)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
