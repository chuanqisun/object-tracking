import asyncio
import json
import statistics
import time
import sys

import cv2
import websockets


async def test(image_path="docker-data/sample.jpg", url="ws://localhost:8765", num_requests=10):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encoding failed")

    image_bytes = encoded.tobytes()
    samples = []

    print(f"Connecting to WebSocket server at {url}...")
    async with websockets.connect(url, max_size=10 * 1024 * 1024) as websocket:
        print("Connected! Sending requests...")
        for i in range(num_requests):
            t0 = time.perf_counter()
            await websocket.send(image_bytes)
            raw_response = await websocket.recv()
            rtt_ms = (time.perf_counter() - t0) * 1000

            res = json.loads(raw_response)
            if res.get("status") != "success":
                raise RuntimeError(f"Server returned error: {res}")

            if i >= 2:  # Skip first 2 warm-up requests
                samples.append(rtt_ms)

            print(
                f"[{i+1:02d}/{num_requests}] "
                f"Detections={res['count']}, "
                f"Inference={res['inference_time_ms']:.2f}ms, "
                f"Server={res['server_time_ms']:.2f}ms, "
                f"RTT={rtt_ms:.2f}ms"
            )

    if samples:
        ordered = sorted(samples)
        p50 = statistics.median(ordered)
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
        print("--------------------------------------------------")
        print(f"Warm-up completed. Warm Median (p50) RTT: {p50:.2f} ms")
        print(f"Warm 95th Percentile (p95) RTT:        {p95:.2f} ms")
        print("--------------------------------------------------")

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "docker-data/sample.jpg"
    asyncio.run(test(image_path=img))
