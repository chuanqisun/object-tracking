#include <iostream>
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <ncnn/gpu.h>

#if __has_include(<turbojpeg.h>)
#include <turbojpeg.h>
#define HAS_TURBOJPEG 1
#else
#define HAS_TURBOJPEG 0
#endif

// Micro HTTP Server using Crow (Header-only: github.com/CrowCpp/Crow)
#include "crow_all.h"

struct DetectionProposal {
    int class_id;
    float score;
    float x1, y1, x2, y2;
};

class YOLO26SegWorker {
private:
    ncnn::Net net;
    int target_size = 640;
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

#if HAS_TURBOJPEG
    tjhandle tj_instance = nullptr;
#endif

public:
    YOLO26SegWorker(const std::string& param_path, const std::string& bin_path) {
#if HAS_TURBOJPEG
        tj_instance = tjInitDecompress();
#endif
        // Tuned for Radeon 890M (RDNA 3.5 / gfx1150)
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
        net.opt.use_packing_layout = true;
        net.opt.use_shader_local_memory = true;
        net.opt.num_threads = 2; // Lightweight feeder per worker on Zen 5

        net.set_vulkan_device(0); // Radeon 890M

        if (net.load_param(param_path.c_str()) != 0 || net.load_model(bin_path.c_str()) != 0) {
            std::cerr << "Failed to load NCNN model files: " << param_path << std::endl;
        }
    }

    ~YOLO26SegWorker() {
        net.clear();
#if HAS_TURBOJPEG
        if (tj_instance) {
            tjDestroy(tj_instance);
        }
#endif
    }

    crow::json::wvalue infer(const unsigned char* data, size_t size) {
        auto start_time = std::chrono::high_resolution_clock::now();
        int img_w = 0;
        int img_h = 0;
        ncnn::Mat in;

        bool decoded = false;

#if HAS_TURBOJPEG
        if (tj_instance && size > 4 && data[0] == 0xFF && data[1] == 0xD8) {
            int subsamp, colorspace;
            if (tjDecompressHeader3(tj_instance, data, size, &img_w, &img_h, &subsamp, &colorspace) == 0) {
                std::vector<uint8_t> rgb_buffer(img_w * img_h * 3);
                if (tjDecompress2(tj_instance, data, size, rgb_buffer.data(),
                                  img_w, 0, img_h, TJPF_RGB, TJFLAG_FASTDCT) == 0) {
                    in = ncnn::Mat::from_pixels_resize(rgb_buffer.data(),
                        ncnn::Mat::PIXEL_RGB, img_w, img_h, target_size, target_size);
                    decoded = true;
                }
            }
        }
#endif

        if (!decoded) {
            cv::Mat raw(1, static_cast<int>(size), CV_8UC1, const_cast<unsigned char*>(data));
            cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
            if (img.empty()) return {{"error", "Invalid image format"}};
            img_w = img.cols;
            img_h = img.rows;
            in = ncnn::Mat::from_pixels_resize(img.data,
                ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, target_size, target_size);
        }

        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in);

        ncnn::Mat out_det;
        ncnn::Mat out_proto;
        ex.extract("out0", out_det);
        ex.extract("out1", out_proto);

        crow::json::wvalue res;
        std::vector<crow::json::wvalue> detections;
        std::vector<DetectionProposal> proposals;
        proposals.reserve(64);

        int num_attrs = (out_det.h == 38 || (out_det.dims == 2 && out_det.h < out_det.w)) ? out_det.h : out_det.w;
        int num_classes = std::max(1, num_attrs - 4 - 32);

        const float score_threshold = 0.25f;
        const float nms_threshold = 0.45f;

        if (out_det.h == 38 || (out_det.dims == 2 && out_det.h < out_det.w)) {
            int num_boxes = out_det.w;
            for (int i = 0; i < num_boxes; i++) {
                float max_score = -1.0f;
                int best_class_id = -1;

                for (int c = 0; c < num_classes; c++) {
                    float s = out_det.row(4 + c)[i];
                    if (s > max_score) {
                        max_score = s;
                        best_class_id = c;
                    }
                }

                if (max_score > score_threshold) {
                    float cx = out_det.row(0)[i];
                    float cy = out_det.row(1)[i];
                    float w  = out_det.row(2)[i];
                    float h  = out_det.row(3)[i];
                    proposals.push_back({best_class_id, max_score, cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f});
                }
            }
        } else {
            int num_boxes = out_det.h;
            for (int i = 0; i < num_boxes; i++) {
                const float* values = out_det.row(i);
                float max_score = -1.0f;
                int best_class_id = -1;

                for (int c = 0; c < num_classes; c++) {
                    float s = values[4 + c];
                    if (s > max_score) {
                        max_score = s;
                        best_class_id = c;
                    }
                }

                if (max_score > score_threshold) {
                    float cx = values[0];
                    float cy = values[1];
                    float w  = values[2];
                    float h  = values[3];
                    proposals.push_back({best_class_id, max_score, cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f});
                }
            }
        }

        // Fast NMS
        std::sort(proposals.begin(), proposals.end(), [](const DetectionProposal& a, const DetectionProposal& b) {
            return a.score > b.score;
        });

        std::vector<DetectionProposal> kept;
        kept.reserve(proposals.size());
        for (const auto& prop : proposals) {
            bool keep = true;
            for (const auto& k : kept) {
                float inter_x1 = std::max(prop.x1, k.x1);
                float inter_y1 = std::max(prop.y1, k.y1);
                float inter_x2 = std::min(prop.x2, k.x2);
                float inter_y2 = std::min(prop.y2, k.y2);
                float inter_w = std::max(0.0f, inter_x2 - inter_x1);
                float inter_h = std::max(0.0f, inter_y2 - inter_y1);
                float inter_area = inter_w * inter_h;
                float area1 = (prop.x2 - prop.x1) * (prop.y2 - prop.y1);
                float area2 = (k.x2 - k.x1) * (k.y2 - k.y1);
                float iou = inter_area / (area1 + area2 - inter_area + 1e-6f);

                if (iou > nms_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                kept.push_back(prop);
            }
        }

        detections.reserve(kept.size());
        for (const auto& obj_det : kept) {
            crow::json::wvalue obj;
            obj["class_id"] = obj_det.class_id;
            obj["score"] = obj_det.score;
            obj["box"] = crow::json::wvalue::list({obj_det.x1, obj_det.y1, obj_det.x2, obj_det.y2});
            detections.push_back(std::move(obj));
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        res["img_w"] = img_w;
        res["img_h"] = img_h;
        res["target_size"] = target_size;
        res["infer_ms"] = infer_ms;
        res["detections"] = std::move(detections);
        return res;
    }
};

// Thread-safe Engine for Single Client Inferencing
class YOLO26SegEngine {
private:
    std::unique_ptr<YOLO26SegWorker> worker;
    std::mutex mtx;

public:
    YOLO26SegEngine(const std::string& param_path, const std::string& bin_path) {
        ncnn::create_gpu_instance();
        worker = std::make_unique<YOLO26SegWorker>(param_path, bin_path);
    }

    ~YOLO26SegEngine() {
        worker.reset();
        ncnn::destroy_gpu_instance();
    }

    crow::json::wvalue infer(const unsigned char* data, size_t len) {
        std::lock_guard<std::mutex> lock(mtx);
        return worker->infer(data, len);
    }
};

int main(int argc, char** argv) {
    std::string param_path = "puck-eye-seg-s_ncnn_model/model.ncnn.param";
    std::string bin_path = "puck-eye-seg-s_ncnn_model/model.ncnn.bin";

    if (argc >= 3) {
        param_path = argv[1];
        bin_path = argv[2];
    } else {
        FILE* fp = fopen(param_path.c_str(), "rb");
        if (fp) {
            fclose(fp);
        } else {
            param_path = "../puck-eye-seg-s_ncnn_model/model.ncnn.param";
            bin_path = "../puck-eye-seg-s_ncnn_model/model.ncnn.bin";
        }
    }

    std::cout << "Loading NCNN model: " << param_path << " and " << bin_path << std::endl;

    crow::SimpleApp app;
    // Single Vulkan worker instance optimized for single client real-time streaming
    YOLO26SegEngine engine(param_path, bin_path);

    // Static Web UI route
    CROW_ROUTE(app, "/")
    ([]() {
        crow::response res;
        std::vector<std::string> candidates = {"web/index.html", "inference/web/index.html", "../inference/web/index.html"};
        for (const auto& path : candidates) {
            FILE* fp = fopen(path.c_str(), "rb");
            if (fp) {
                fclose(fp);
                res.set_static_file_info(path);
                return res;
            }
        }
        res.code = 404;
        res.write("index.html not found");
        return res;
    });

    // WebSocket endpoint for real-time video stream inferencing
    CROW_WEBSOCKET_ROUTE(app, "/ws")
    .onopen([&](crow::websocket::connection& conn) {
        std::cout << "[WebSocket] Client connected: " << conn.get_remote_ip() << std::endl;
    })
    .onclose([&](crow::websocket::connection& conn, const std::string& reason) {
        std::cout << "[WebSocket] Client disconnected: " << reason << std::endl;
    })
    .onerror([&](crow::websocket::connection& conn, const std::string& error) {
        std::cerr << "[WebSocket] Error: " << error << std::endl;
    })
    .onmessage([&engine](crow::websocket::connection& conn, const std::string& data, bool /*is_binary*/) {
        if (data.empty()) return;
        auto result = engine.infer(reinterpret_cast<const unsigned char*>(data.data()), data.size());
        conn.send_text(result.dump());
    });

    // HTTP POST endpoint
    CROW_ROUTE(app, "/predict").methods("POST"_method, "OPTIONS"_method)
    ([&engine](const crow::request& req) {
        if (req.method == "OPTIONS"_method) {
            crow::response res(200);
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
            return res;
        }
        auto result = engine.infer(reinterpret_cast<const unsigned char*>(req.body.data()), req.body.size());
        crow::response res(result);
        res.set_header("Access-Control-Allow-Origin", "*");
        return res;
    });

    int port = (argc >= 4) ? std::atoi(argv[3]) : 18888;
    std::cout << "Server starting on http://localhost:" << port << " (WebSocket at ws://localhost:" << port << "/ws)" << std::endl;
    try {
        app.port(port).multithreaded().run();
    } catch (const std::exception& e) {
        std::cerr << "Error starting server on port " << port << ": " << e.what() << std::endl;
        return 1;
    }
}