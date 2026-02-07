#include "detect.h"
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir> <model_path>" << std::endl;
        return -1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];
    std::string model_path = argv[3];

    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    // Default to RECT mode as specified in detect.h
    yolo::Detector detector(0.25f, 0.45f, 640);
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
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".JPG" || ext == ".PNG") {
                cv::Mat frame = cv::imread(path);
                if (frame.empty()) {
                    std::cerr << "Failed to read image: " << path << std::endl;
                    continue;
                }

                std::vector<yolo::Detector::Result> results;
                if (detector.Detect(frame, results) == yolo::SUCCESS) {
                    cv::Mat canvas = frame.clone();
                    for (const auto& res : results) {
                        cv::Scalar color = colors[res.class_id % colors.size()];

                        // Draw semi-transparent mask
                        if (!res.mask.empty()) {
                            cv::Mat mask_colored = cv::Mat::zeros(frame.size(), frame.type());
                            mask_colored.setTo(color, res.mask);
                            cv::addWeighted(canvas, 1.0, mask_colored, 0.4, 0, canvas);
                        }

                        // Draw bounding box
                        cv::rectangle(canvas, res.box, color, 2);

                        // Draw label and confidence
                        std::string label = "Class " + std::to_string(res.class_id) + " " + std::to_string(static_cast<int>(res.confidence * 100)) + "%";
                        int baseLine;
                        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                        cv::rectangle(canvas, cv::Rect(res.box.x, res.box.y - labelSize.height - 5, labelSize.width, labelSize.height + 5), color, cv::FILLED);
                        cv::putText(canvas, label, cv::Point(res.box.x, res.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                    }

                    std::string filename = entry.path().filename().string();
                    std::string out_path = (fs::path(output_dir) / filename).string();
                    cv::imwrite(out_path, canvas);
                    std::cout << "Processed: " << path << " -> " << out_path << std::endl;
                } else {
                    std::cerr << "Inference failed for: " << path << std::endl;
                }
            }
        }
    }

    std::cout << "Batch processing complete." << std::endl;
    return 0;
}
