#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <ncnn/gpu.h>

// Micro HTTP Server using Crow (Header-only: github.com/CrowCpp/Crow)
#include "crow_all.h" 

class YOLO26SegEngine {
private:
    ncnn::Net net;
    int target_size = 640;
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

public:
    YOLO26SegEngine(const std::string& param_path, const std::string& bin_path) {
        // 1. Enable Vulkan GPU Acceleration
        ncnn::create_gpu_instance();
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
        net.opt.num_threads = 4; // Lightweight CPU thread feeder

        // 2. Select Radeon 890M GPU (Device 0)
        net.set_vulkan_device(0);

        // 3. Load Model
        if (net.load_param(param_path.c_str()) != 0 || net.load_model(bin_path.c_str()) != 0) {
            std::cerr << "Failed to load NCNN model files!" << std::endl;
        }
    }

    ~YOLO26SegEngine() {
        net.clear();
        ncnn::destroy_gpu_instance();
    }

    crow::json::wvalue infer(const std::vector<uchar>& image_bytes) {
        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat img = cv::imdecode(image_bytes, cv::IMREAD_COLOR);
        if (img.empty()) return {{"error", "Invalid image"}};

        int img_w = img.cols;
        int img_h = img.rows;

        // Resize / Letterbox to 640x640
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, 
            ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, target_size, target_size);
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in); // Input blob name is "in0"

        ncnn::Mat out_det;
        ncnn::Mat out_proto;
        
        // YOLO26 End-to-End NMS-Free Head Output + Segmentation Prototypes
        ex.extract("out0", out_det);
        ex.extract("out1", out_proto);

        // Parse Detections and Masks
        crow::json::wvalue res;
        std::vector<crow::json::wvalue> detections;

        // In NCNN, out0 shape is either (38, num_boxes) or (num_boxes, 38)
        int num_boxes = (out_det.h > 1) ? out_det.h : out_det.w;
        int num_attrs = (out_det.h > 1) ? out_det.w : out_det.h;

        float score_threshold = 0.25f;

        if (out_det.h == 38 || (out_det.dims == 2 && out_det.h < out_det.w)) {
            // Shape is (38, num_boxes) - transposed layout: channel/attribute is row
            for (int i = 0; i < out_det.w; i++) {
                float x1 = out_det.row(0)[i];
                float y1 = out_det.row(1)[i];
                float x2 = out_det.row(2)[i];
                float y2 = out_det.row(3)[i];
                float score = out_det.row(4)[i];
                int class_id = (int)out_det.row(5)[i];

                if (score > score_threshold) {
                    crow::json::wvalue obj;
                    obj["class_id"] = class_id;
                    obj["score"] = score;
                    obj["box"] = crow::json::wvalue::list({x1, y1, x2, y2});
                    detections.push_back(obj);
                }
            }
        } else {
            // Shape is (num_boxes, 38) - standard layout: each row is a box
            for (int i = 0; i < out_det.h; i++) {
                const float* values = out_det.row(i);
                float score = values[4];
                if (score > score_threshold) {
                    crow::json::wvalue obj;
                    obj["class_id"] = (int)values[5];
                    obj["score"] = score;
                    obj["box"] = crow::json::wvalue::list({values[0], values[1], values[2], values[3]});
                    detections.push_back(obj);
                }
            }
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
    .onmessage([&engine](crow::websocket::connection& conn, const std::string& data, bool is_binary) {
        if (data.empty()) return;
        std::vector<uchar> image_bytes(data.begin(), data.end());
        auto result = engine.infer(image_bytes);
        conn.send_text(result.dump());
    });

    // HTTP POST fallback endpoint
    CROW_ROUTE(app, "/predict").methods("POST"_method, "OPTIONS"_method)
    ([&engine](const crow::request& req) {
        if (req.method == "OPTIONS"_method) {
            crow::response res(200);
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
            return res;
        }
        std::vector<uchar> image_bytes(req.body.begin(), req.body.end());
        auto result = engine.infer(image_bytes);
        crow::response res(result);
        res.set_header("Access-Control-Allow-Origin", "*");
        return res;
    });

    int port = (argc >= 4) ? std::atoi(argv[3]) : 18888;
    std::cout << "Server starting on http://localhost:" << port << " (WebSocket at ws://localhost:" << port << "/ws)" << std::endl;
    app.port(port).multithreaded().run();
}