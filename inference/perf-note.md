## Diagnosis

Your 19 FPS is probably **not caused by NMS or JSON**. The main issues are:

### 1. You compute and download the segmentation output but never use it

```cpp
ex.extract("out0", out_det);
ex.extract("out1", out_proto);
```

`out_proto` is discarded. It is typically about `32×160×160`, so the second extraction executes/downloads a substantial branch and introduces another synchronization point.

If you only need boxes, remove it:

```cpp
ex.extract("out0", out_det);
// ex.extract("out1", out_proto);  // Remove
```

Better still, export a **YOLO26 detection model**, not a segmentation model. Otherwise the shared segmentation head may still consume GPU compute.

If masks are required, your current program is incomplete: it never retains mask coefficients or combines them with `out_proto`.

---

### 2. The pipeline is completely serialized

Your timing is effectively:

```text
JPEG decode
→ CPU resize
→ CPU normalization
→ GPU upload
→ GPU inference
→ output download/synchronization
→ CPU postprocess
→ JSON serialization
→ next frame
```

This mutex guarantees only one request is processed at a time:

```cpp
std::lock_guard<std::mutex> lock(mtx);
```

Consequently, JPEG decoding and post-processing leave the GPU idle. At 19 FPS, total pipeline latency is approximately **52.6 ms**, but that does not reveal how much is actual model execution.

Use three stages:

1. Decode/preprocess frame `N+1`
2. Infer frame `N`
3. Postprocess/send frame `N-1`

Because concurrent extractors sharing one `Net`/Vulkan resources may require careful allocator handling, start with one inference thread and separate decode/postprocess threads.

---

### 3. You have no warm-up or component-level timing

NCNN’s own benchmark performs several warm-up iterations and configures reusable CPU/Vulkan allocators. ([github.com](https://github.com/Tencent/ncnn/blob/master/benchmark/benchncnn.cpp?utm_source=openai))

Measure these separately:

```cpp
decode_ms
preprocess_ms
extract_out0_ms
extract_out1_ms
postprocess_ms
json_ms
total_ms
```

Add at least 10 warm-up runs before reporting performance.

A quick experiment:

```cpp
auto t0 = clock::now();
// decode
auto t1 = clock::now();
// resize + normalize
auto t2 = clock::now();
ex.input("in0", in);
ex.extract("out0", out_det);
auto t3 = clock::now();
// postprocessing
auto t4 = clock::now();
```

This will immediately show whether the 19 FPS is model-bound or pipeline-bound.

---

### 4. Vulkan memory allocators are not explicitly reused

NCNN recommends assigning Vulkan blob/workspace/staging allocators. Its benchmark does this as well. ([github.com](https://github.com/Tencent/ncnn/blob/master/benchmark/benchncnn.cpp?utm_source=openai))

Add persistent allocators to the worker:

```cpp
ncnn::VulkanDevice* vkdev = nullptr;
ncnn::VkAllocator* blob_vkallocator = nullptr;
ncnn::VkAllocator* staging_vkallocator = nullptr;
```

Initialize them:

```cpp
vkdev = ncnn::get_gpu_device(0);

blob_vkallocator = vkdev->acquire_blob_allocator();
staging_vkallocator = vkdev->acquire_staging_allocator();

net.opt.blob_vkallocator = blob_vkallocator;
net.opt.workspace_vkallocator = blob_vkallocator;
net.opt.staging_vkallocator = staging_vkallocator;
```

Release them after `net.clear()`:

```cpp
vkdev->reclaim_blob_allocator(blob_vkallocator);
vkdev->reclaim_staging_allocator(staging_vkallocator);
```

Also enable:

```cpp
net.opt.lightmode = true;
net.opt.use_winograd_convolution = true;
net.opt.use_sgemm_convolution = true;
```

---

### 5. Per-frame allocations add avoidable overhead

This allocates a new RGB buffer for every JPEG:

```cpp
std::vector<uint8_t> rgb_buffer(img_w * img_h * 3);
```

Make it a worker member:

```cpp
std::vector<uint8_t> rgb_buffer;
```

Then:

```cpp
rgb_buffer.resize(static_cast<size_t>(img_w) * img_h * 3);
```

Also reuse:

- `proposals`
- `kept`
- preprocessing buffers
- Crow response structures where practical

This likely yields only a modest improvement, but stabilizes latency.

---

### 6. Your model-output interpretation is fragile

This condition is unsafe:

```cpp
out_det.h == 38
```

For segmentation, 38 commonly means:

```text
4 coordinates + 2 classes + 32 mask coefficients
```

But end-to-end YOLO26 segmentation can also use 38 as:

```text
4 xyxy + confidence + class ID + 32 coefficients
```

NCNN exports currently fall back to the traditional one-to-many output, which requires NMS, but you should verify the exact exported tensor layout rather than infer it from dimensions. ([docs.ultralytics.com](https://docs.ultralytics.com/guides/end2end-detection?utm_source=openai))

Print it once:

```cpp
std::cerr
    << "out0 dims=" << out_det.dims
    << " w=" << out_det.w
    << " h=" << out_det.h
    << " c=" << out_det.c
    << " d=" << out_det.d
    << '\n';
```

Inspect the `.param` output layers and compare results against Ultralytics on the same image.

---

### 7. Mask coefficients are ignored

For traditional segmentation output:

```text
[box coordinates][class scores][32 mask coefficients]
```

Your code reads boxes/classes but discards the coefficients. Therefore this is functionally a detector that pays for a segmentation model.

Choose one:

- **Boxes only:** export/train a detection model.
- **Masks needed:** retain 32 coefficients for each proposal, then reconstruct masks from `out_proto`.
- **Occasional masks:** run detection every frame and segmentation only on selected frames.

The last approach can significantly raise displayed FPS.

---

### 8. NMS incorrectly suppresses boxes across classes

Your NMS is class-agnostic:

```cpp
for (const auto& k : kept) {
    if (iou > nms_threshold)
        keep = false;
}
```

Use:

```cpp
if (prop.class_id != k.class_id)
    continue;
```

unless class-agnostic NMS is intentional.

Also cap candidates before quadratic NMS:

```cpp
constexpr size_t max_nms_candidates = 300;
if (proposals.size() > max_nms_candidates)
    proposals.resize(max_nms_candidates);
```

---

### 9. Resize distorts the input and coordinates are not mapped back

You resize directly from the original aspect ratio to `640×640`:

```cpp
from_pixels_resize(..., target_size, target_size);
```

That distorts the image. Returned boxes also remain in 640×640 model coordinates despite reporting the original dimensions.

Use letterboxing and reverse the scale/padding:

```cpp
scale = std::min(640.f / img_w, 640.f / img_h);
new_w = round(img_w * scale);
new_h = round(img_h * scale);
pad_x = (640 - new_w) / 2;
pad_y = (640 - new_h) / 2;
```

Then map results back:

```cpp
x = (x - pad_x) / scale;
y = (y - pad_y) / scale;
```

This is primarily a correctness issue.

---

## Highest-value experiments

Run these one at a time:

1. **Remove `out1` extraction.**
2. Benchmark with a pre-created random `640×640` input—no JPEG/server.
3. Compare GPU versus CPU:
   ```cpp
   net.opt.use_vulkan_compute = false;
   net.opt.num_threads = 8;
   ```
4. Try input sizes `640`, `512`, and `480`.
5. Export an equivalent detection-only model.
6. Add persistent Vulkan allocators.
7. Pipeline JPEG decoding with inference.

Expected scaling from resolution alone:

| Input | Approximate compute vs 640 |
|---|---:|
| 640 | 100% |
| 512 | 64% |
| 480 | 56% |
| 416 | 42% |

Thus, if inference is compute-bound, 19 FPS at 640 could become approximately 28–32 FPS at 512, subject to layer and bandwidth behavior.

## Bottom line

The biggest problem is architectural: **you run a segmentation model, extract its large prototype output, then discard every mask**. First remove `out1` and measure pure `out0` inference. If boxes alone are sufficient, switching to a detection-only export is likely the largest improvement. After that, pipeline decode/inference/postprocess so the GPU is not idle between frames.

## References

1. [ncnn/benchmark/benchncnn.cpp at master · Tencent/ncnn · GitHub](https://github.com/Tencent/ncnn/blob/master/benchmark/benchncnn.cpp?utm_source=openai)
2. [YOLO26 End-to-End NMS-Free Detection | Ultralytics](https://docs.ultralytics.com/guides/end2end-detection?utm_source=openai)
