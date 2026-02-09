// detect.h
#ifndef YOLO_INFERENCE_H_
#define YOLO_INFERENCE_H_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace yolo
{
    enum StatusCode
    {
        SUCCESS = 0,

        INFERENCE_ERROR = -0X0001,
        UNKNOWN_ERROR = -0X8000,
    };
    class Detector
    {
    public:
        struct Result
        {
            size_t class_id;
            float confidence;
            cv::Rect box;
            cv::Mat mask;
            Result(size_t cid, float conf, cv::Rect b) : class_id(cid), confidence(conf), box(b), mask() {}
        };
        enum Order
        {
            AREA_LARGE,
            AREA_SMALL,
            X_LEFT,
            X_RIGHT,
            Y_TOP,
            Y_BOTTOM,

            NONE,
        };
        enum imageClass
        {
            BACKGROUND = 0,
            PHOTO = 1,
            LAYOUT = 2,
        };

        enum class PreprocessingMethod
        {
            LETTERBOX,
            RESIZE,
            RECT,
        };

        Detector(float confidence_threshold, float NMS_threshold, int imgsz, PreprocessingMethod method = PreprocessingMethod::RECT);
        Detector(float confidence_threshold, float NMS_threshold, cv::Size input_shape = cv::Size(640, 640), PreprocessingMethod method = PreprocessingMethod::RECT);
        ~Detector() = default;

        int InitVino(const std::string &model_path);

        StatusCode Detect(const cv::Mat &frame, std::vector<Result> &results, Order order = Order::NONE);

    private:
        void preProcessing(const cv::Mat &frame);
        std::vector<Result> postProcessing(const cv::Mat &frame, Order order = Order::NONE);
        cv::Rect scaleBoundingBox(const cv::Rect2d &src) const;

        void setupModelOutputShape(std::shared_ptr<const ov::Model> model);

        ov::Core ov_core;
        ov::InferRequest m_inference_request;
        ov::CompiledModel m_compiled_model;

        cv::Size m_model_input_shape;
        cv::Point2i m_pad;
        cv::Rect2d m_input_resized_roi;
        cv::Size m_model_output_shape_det;
        cv::Size m_model_output_shape_seg;
        cv::Point2f m_scale_factor;
        int m_segment_channel{32};
        static constexpr int m_stride = 32;

        cv::Mat m_resized_frame;
        cv::Mat m_input_blob;
        ov::Tensor m_input_tensor;

        float m_confidence_threshold;
        float m_NMS_threshold;
        PreprocessingMethod m_preprocessing_method;
    };

} // namespace yolo

#endif // YOLO_INFERENCE_H_