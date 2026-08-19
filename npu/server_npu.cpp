#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <sched.h>
#include <pthread.h>

// Networking & Hardware
#include <App.h>
#include <turbojpeg.h>
#include <onnxruntime_cxx_api.h>

#pragma pack(push, 1)
struct DetectionWire {
    int32_t class_id;
    float score;
    float x1, y1, x2, y2;
    uint32_t mask_offset; // Byte offset into trailing payload
    uint32_t mask_size;   // Size of mask bytes (160x160 binary mask)
};

struct ResponseHeader {
    uint32_t frame_id;
    float infer_ms;
    uint32_t num_detections;
};
#pragma pack(pop)

struct FrameSlot {
    std::vector<uint8_t> buffer;
    size_t size = 0;
    uint32_t frame_id = 0;
    void* ws_ptr = nullptr;
    std::atomic<bool> ready{false};
};

class LowLatencyNPUWorker {
private:
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "BF16_NPU_Server"};
    Ort::SessionOptions session_opts;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const char* input_names[1] = {"images"};
    const char* output_names[2] = {"out0", "out1"}; // [0]: Detections + Mask Coeffs, [1]: Proto Masks
    const int64_t in_shape[4] = {1, 3, 640, 640};

    // Preallocated buffers
    std::vector<float> input_tensor;
    std::vector<uint8_t> rgb_raw;
    std::vector<float> mask_coeffs_buffer;
    tjhandle tj_instance = nullptr;

    inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

public:
    LowLatencyNPUWorker(const std::string& compiled_model_path) {
        input_tensor.resize(1 * 3 * 640 * 640);
        rgb_raw.resize(3840 * 2160 * 3); // Max input image buffer
        mask_coeffs_buffer.resize(32);
        tj_instance = tjInitDecompress();

        session_opts.SetIntraOpNumThreads(4);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Vitis-AI Execution Provider Options
        std::unordered_map<std::string, std::string> vitis_opts = {
            {"config_file", "vaip_bf16_config.json"},
            {"cacheDir", "./npu_cache_bf16"},
            {"cacheKey", "puck-eye-seg-s-bf16"}
        };

        std::cout << "[INFO] Requesting VitisAIExecutionProvider for native NPU execution..." << std::endl;
        try {
            session_opts.AppendExecutionProvider("VitisAIExecutionProvider", vitis_opts);
            session = std::make_unique<Ort::Session>(env, compiled_model_path.c_str(), session_opts);
            std::cout << "[INFO] Native XDNA 2 NPU Session Loaded Successfully." << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "[FATAL] Failed to initialize VitisAIExecutionProvider: " << e.what() << std::endl;
            throw;
        } catch (const std::exception& e) {
            std::cerr << "[FATAL] Unexpected error initializing VitisAIExecutionProvider: " << e.what() << std::endl;
            throw;
        }
    }

    ~LowLatencyNPUWorker() {
        if (tj_instance) tjDestroy(tj_instance);
    }

    void process(const uint8_t* jpeg_data, size_t size, std::vector<uint8_t>& out_payload, float& infer_time_ms) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. SIMD Decompress to contiguous RGB
        int orig_w, orig_h, subsamp, cs;
        if (tjDecompressHeader3(tj_instance, jpeg_data, size, &orig_w, &orig_h, &subsamp, &cs) != 0) return;
        tjDecompress2(tj_instance, jpeg_data, size, rgb_raw.data(), orig_w, 0, orig_h, TJPF_RGB, TJFLAG_FASTDCT);

        // 2. Direct Planar Normalization (AVX-512 targetable)
        float* ptr_r = input_tensor.data();
        float* ptr_g = ptr_r + (640 * 640);
        float* ptr_b = ptr_g + (640 * 640);

        float x_ratio = (float)orig_w / 640.0f;
        float y_ratio = (float)orig_h / 640.0f;

        for (int y = 0; y < 640; ++y) {
            int src_y = static_cast<int>(y * y_ratio);
            const uint8_t* row = &rgb_raw[src_y * orig_w * 3];
            int row_idx = y * 640;
            for (int x = 0; x < 640; ++x) {
                int src_x = static_cast<int>(x * x_ratio) * 3;
                ptr_r[row_idx + x] = row[src_x] * (1.0f / 255.0f);
                ptr_g[row_idx + x] = row[src_x + 1] * (1.0f / 255.0f);
                ptr_b[row_idx + x] = row[src_x + 2] * (1.0f / 255.0f);
            }
        }

        // 3. Hardware BF16 Execution on AMD XDNA 2
        Ort::Value in_val = Ort::Value::CreateTensor<float>(mem_info, input_tensor.data(), input_tensor.size(), in_shape, 4);
        auto out_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &in_val, 1, output_names, 2);

        float* det_data = out_tensors[0].GetTensorMutableData<float>();
        auto det_shape = out_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_anchors = (det_shape[1] == 38 || det_shape[1] < det_shape[2]) ? det_shape[2] : det_shape[1];
        int num_classes = (det_shape[1] == 38 || det_shape[1] < det_shape[2]) ? (det_shape[1] - 4 - 32) : (det_shape[2] - 4 - 32);

        float* proto_data = (out_tensors.size() > 1) ? out_tensors[1].GetTensorMutableData<float>() : nullptr;

        // 4. Candidate Extraction
        struct Cand {
            int cls;
            float score;
            float box[4];
            std::vector<float> coeffs;
        };
        std::vector<Cand> candidates;

        for (int i = 0; i < num_anchors; ++i) {
            float max_s = -1.0f;
            int best_cls = -1;
            for (int c = 0; c < num_classes; ++c) {
                float s = det_data[(4 + c) * num_anchors + i];
                if (s > max_s) { max_s = s; best_cls = c; }
            }

            if (max_s > 0.25f) {
                float cx = det_data[0 * num_anchors + i] * x_ratio;
                float cy = det_data[1 * num_anchors + i] * y_ratio;
                float w  = det_data[2 * num_anchors + i] * x_ratio;
                float h  = det_data[3 * num_anchors + i] * y_ratio;

                std::vector<float> coeffs(32);
                for (int m = 0; m < 32; ++m) {
                    coeffs[m] = det_data[(4 + num_classes + m) * num_anchors + i];
                }
                candidates.push_back({best_cls, max_s, {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f}, std::move(coeffs)});
            }
        }

        // 5. Fast NMS
        std::sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
        std::vector<Cand> kept;
        for (const auto& d : candidates) {
            bool keep = true;
            for (const auto& k : kept) {
                float ix1 = std::max(d.box[0], k.box[0]), iy1 = std::max(d.box[1], k.box[1]);
                float ix2 = std::min(d.box[2], k.box[2]), iy2 = std::min(d.box[3], k.box[3]);
                float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
                float inter = iw * ih;
                float a1 = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]);
                float a2 = (k.box[2] - k.box[0]) * (k.box[3] - k.box[1]);
                if (inter / (a1 + a2 - inter + 1e-6f) > 0.45f) { keep = false; break; }
            }
            if (keep) kept.push_back(d);
            if (kept.size() >= 10) break; // Hard limit for maximum realtime stream throughput
        }

        // 6. Vectorized Prototype Mask GEMM (160x160 mask generation)
        std::vector<uint8_t> mask_bytes_stream;
        std::vector<DetectionWire> wire_dets;

        const int proto_hw = 160 * 160;
        for (const auto& obj : kept) {
            uint32_t cur_offset = static_cast<uint32_t>(mask_bytes_stream.size());
            if (proto_data) {
                for (int p = 0; p < proto_hw; ++p) {
                    float logit = 0.0f;
                    for (int c = 0; c < 32; ++c) {
                        logit += obj.coeffs[c] * proto_data[c * proto_hw + p];
                    }
                    mask_bytes_stream.push_back(sigmoid(logit) > 0.5f ? 255 : 0);
                }
            }
            wire_dets.push_back({
                obj.cls,
                obj.score,
                obj.box[0], obj.box[1], obj.box[2], obj.box[3],
                cur_offset,
                proto_data ? static_cast<uint32_t>(proto_hw) : 0u
            });
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        infer_time_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // 7. Binary Protocol Packet Assembly
        ResponseHeader header{0, infer_time_ms, static_cast<uint32_t>(wire_dets.size())};
        size_t total_size = sizeof(ResponseHeader) + (wire_dets.size() * sizeof(DetectionWire)) + mask_bytes_stream.size();
        out_payload.resize(total_size);

        uint8_t* out_ptr = out_payload.data();
        std::memcpy(out_ptr, &header, sizeof(ResponseHeader));
        out_ptr += sizeof(ResponseHeader);

        if (!wire_dets.empty()) {
            size_t dets_size = wire_dets.size() * sizeof(DetectionWire);
            std::memcpy(out_ptr, wire_dets.data(), dets_size);
            out_ptr += dets_size;
        }

        if (!mask_bytes_stream.empty()) {
            std::memcpy(out_ptr, mask_bytes_stream.data(), mask_bytes_stream.size());
        }
    }
};

// Global Lock-Free Single-Slot Buffer
FrameSlot g_frame_slot;
std::mutex g_slot_mtx;
std::condition_variable g_slot_cv;
std::atomic<bool> g_running{true};
uWS::Loop* g_loop = nullptr;
std::unordered_set<void*> g_active_sockets;

struct UserData {};

int main(int argc, char** argv) {
    std::string model_path = (argc > 1) ? argv[1] : "puck-eye-seg-s_bf16_ctx.onnx";
    int port = (argc > 2) ? std::stoi(argv[2]) : 18888;

    g_frame_slot.buffer.resize(4 * 1024 * 1024);

    // Dedicated NPU Worker Thread pinned to Core 2
    std::thread worker_thread([&]() {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(2, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        try {
            LowLatencyNPUWorker worker(model_path);
            std::vector<uint8_t> response_payload;
            std::vector<uint8_t> local_frame(4 * 1024 * 1024);

            while (g_running) {
                size_t frame_size = 0;
                uWS::WebSocket<false, true, UserData>* target_ws = nullptr;

                {
                    std::unique_lock<std::mutex> lock(g_slot_mtx);
                    g_slot_cv.wait(lock, [] { return g_frame_slot.ready.load() || !g_running; });
                    if (!g_running) break;

                    frame_size = g_frame_slot.size;
                    target_ws = static_cast<uWS::WebSocket<false, true, UserData>*>(g_frame_slot.ws_ptr);
                    std::memcpy(local_frame.data(), g_frame_slot.buffer.data(), frame_size);
                    g_frame_slot.ready.store(false);
                }

                float infer_time = 0.0f;
                worker.process(local_frame.data(), frame_size, response_payload, infer_time);

                if (target_ws && g_loop) {
                    g_loop->defer([target_ws, payload = std::move(response_payload)]() {
                        if (g_active_sockets.count(target_ws)) {
                            target_ws->send(std::string_view(reinterpret_cast<const char*>(payload.data()), payload.size()), uWS::OpCode::BINARY);
                        }
                    });
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[FATAL] Worker thread error: " << e.what() << std::endl;
            g_running = false;
        }
    });

    // Network Thread pinned to Core 0
    cpu_set_t net_cpuset;
    CPU_ZERO(&net_cpuset);
    CPU_SET(0, &net_cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &net_cpuset);

    g_loop = uWS::Loop::get();

    uWS::App()
        .ws<UserData>("/*", {
            .compression = uWS::DISABLED,
            .maxPayloadLength = 16 * 1024 * 1024,
            .idleTimeout = 120,
            .open = [](auto* ws) {
                g_active_sockets.insert(ws);
                std::cout << "[uWS] Client Connected" << std::endl;
            },
            .message = [](auto* ws, std::string_view message, uWS::OpCode opCode) {
                if (opCode != uWS::OpCode::BINARY || message.empty()) return;

                {
                    std::lock_guard<std::mutex> lock(g_slot_mtx);
                    if (message.size() <= g_frame_slot.buffer.size()) {
                        std::memcpy(g_frame_slot.buffer.data(), message.data(), message.size());
                        g_frame_slot.size = message.size();
                        g_frame_slot.ws_ptr = ws;
                        g_frame_slot.ready.store(true);
                    }
                }
                g_slot_cv.notify_one();
            },
            .close = [](auto* ws, int, std::string_view) {
                g_active_sockets.erase(ws);
                {
                    std::lock_guard<std::mutex> lock(g_slot_mtx);
                    if (g_frame_slot.ws_ptr == ws) {
                        g_frame_slot.ws_ptr = nullptr;
                    }
                }
                std::cout << "[uWS] Client Disconnected" << std::endl;
            }
        })
        .listen(port, [port](auto* listen_socket) {
            if (listen_socket) {
                std::cout << "==========================================================" << std::endl;
                std::cout << " Server Online: ws://0.0.0.0:" << port << " [AMD NPU BF16]" << std::endl;
                std::cout << "==========================================================" << std::endl;
            }
        })
        .run();

    g_running = false;
    g_slot_cv.notify_all();
    if (worker_thread.joinable()) worker_thread.join();
    return 0;
}