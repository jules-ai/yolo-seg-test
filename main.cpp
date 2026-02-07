#include "detect.h"
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <input_dir> <output_dir>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string input_dir = argv[2];
    std::string output_dir = argv[3];

    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    // Default parameters: 0.45 confidence, 0.6 NMS, 1280 imgsz, RECT mode
    yolo::Detector detector(0.45f, 0.6f, 1280);
    if (detector.InitVino(model_path) != yolo::SUCCESS) {
        std::cerr << "Failed to initialize detector with model: " << model_path << std::endl;
        return -1;
    }

    // Different colors for different classes
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),   // Class 0 - Blue
        cv::Scalar(0, 255, 0),   // Class 1 - Green
        cv::Scalar(0, 0, 255),   // Class 2 - Red
        cv::Scalar(255, 255, 0), // Class 3 - Cyan
        cv::Scalar(255, 0, 255), // Class 4 - Magenta
        cv::Scalar(0, 255, 255)  // Class 5 - Yellow
    };

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp") {
                cv::Mat frame = cv::imread(path);
                if (frame.empty()) {
                    std::cerr << "Failed to read image: " << path << std::endl;
                    continue;
                }

                std::vector<yolo::Detector::Result> results;
                if (detector.Detect(frame, results) == yolo::SUCCESS) {
                    cv::Mat canvas = frame.clone();

                    int thickness = std::max(1, static_cast<int>(frame.cols / 300));
                    double font_scale = frame.cols / 1000.0;

                    int photo_count = 0;
                    int layout_count = 0;

                    for (const auto& res : results) {
                        if (res.class_id == yolo::Detector::PHOTO) photo_count++;
                        else if (res.class_id == yolo::Detector::LAYOUT) layout_count++;

                        cv::Scalar color = colors[res.class_id % colors.size()];

                        // Draw semi-transparent mask
                        if (!res.mask.empty()) {
                            cv::Mat mask_colored = cv::Mat::zeros(frame.size(), frame.type());
                            mask_colored.setTo(color, res.mask);
                            cv::addWeighted(canvas, 1.0, mask_colored, 0.4, 0, canvas);
                        }

                        // Draw bounding box
                        cv::rectangle(canvas, res.box, color, thickness);

                        // Draw label and confidence
                        std::string label = "Class " + std::to_string(res.class_id) + " " + std::to_string(static_cast<int>(res.confidence * 100)) + "%";
                        int baseLine;
                        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, std::max(1, thickness/2), &baseLine);
                        cv::rectangle(canvas, cv::Rect(res.box.x, res.box.y - labelSize.height - 5, labelSize.width, labelSize.height + 5), color, cv::FILLED);
                        cv::putText(canvas, label, cv::Point(res.box.x, res.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), std::max(1, thickness/2));
                    }

                    std::string filename = entry.path().filename().string();
                    std::string out_path = (fs::path(output_dir) / filename).string();
                    cv::imwrite(out_path, canvas);
                    std::cout << "Processed: [P" << photo_count << " L" << layout_count << "] " << path << " -> " << out_path << std::endl;
                } else {
                    std::cerr << "Inference failed for: " << path << std::endl;
                }
            }
        }
    }

    std::cout << "Batch processing complete." << std::endl;
    return 0;
}
