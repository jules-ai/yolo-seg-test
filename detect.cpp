#include "detect.h"
#include "dencryptor.h"
#include <memory>
#include <random>
#include <iostream>
#include <vector>
#include <fstream>
#include "macro_def.h"

#include "auxi_funcs.hpp"
#include <blake3.h>
#include <filesystem>
#ifdef _WIN32
namespace fs = std::filesystem;
#else
namespace fs = std::__fs::filesystem;
#endif
namespace yolo
{
    static std::string hash(const char *data, size_t size)
    {
        blake3_hasher hasher;
        uint8_t output[BLAKE3_OUT_LEN];
        blake3_hasher_init(&hasher);
        blake3_hasher_update(&hasher, data, size);
        blake3_hasher_finalize(&hasher, output, BLAKE3_OUT_LEN);
        static constexpr char hex[] = "0123456789abcdef";
        std::string result(BLAKE3_OUT_LEN, '\0');
        for (size_t i = 0; i < BLAKE3_OUT_LEN / 2; i++)
        {
            result[2 * i] = hex[output[i] >> 4];
            result[2 * i + 1] = hex[output[i] & 0x0F];
        }
        return result;
    }
    int Detector::InitVino(const std::string &model_path)
    {
        std::shared_ptr<ov::Model> model = ov_core.read_model(model_path);

        try
        {
            m_compiled_model = ov_core.compile_model(model, "CPU");
            m_inference_request = m_compiled_model.create_infer_request();
        }
        catch (const ov::Exception &e)
        {
            std::cerr << "Caught ov::Exception : " << e.what() << std::endl;
            return UNKNOWN_ERROR;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Caught const std::exception : " << e.what() << std::endl;
            return UNKNOWN_ERROR;
        }
        catch (...)
        {
            std::cerr << "Caught Unknown exception " << std::endl;
            return UNKNOWN_ERROR;
        }

        // Get input shape from the model
        const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
        const ov::Shape input_shape = inputs[0].get_shape();
        auto height = input_shape[2];
        auto width = input_shape[3];
        m_model_input_shape = cv::Size(static_cast<int>(width), static_cast<int>(height));

        setupModelOutputShape(model);
        return SUCCESS;
    }

    Detector::Detector(const float confidence_threshold, const float NMS_threshold, const cv::Size input_shape)
    {
        m_confidence_threshold = confidence_threshold;
        m_NMS_threshold = NMS_threshold;
        m_model_input_shape = input_shape;
    }
    StatusCode Detector::Detect(const cv::Mat &frame, std::vector<Result> &results, Order order /*= Order::NONE*/)
    {
        preProcessing(frame); // Preprocess the input frame
        try
        {
            m_inference_request.infer();
        }
        catch (const ov::Exception &e)
        {
            std::cerr << "Caught ov::Exception : " << e.what() << std::endl;
            return StatusCode::INFERENCE_ERROR;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Caught std::exception : " << e.what() << std::endl;
            return StatusCode::INFERENCE_ERROR;
        }
        catch (...)
        {
            std::cerr << "Caught Unknown exception " << std::endl;
            return StatusCode::INFERENCE_ERROR;
        }
        results = postProcessing(frame, order);
        return StatusCode::SUCCESS;
    }

    // Method to preprocess the input frame
    void Detector::preProcessing(const cv::Mat &frame)
    {
        if (m_model_input_shape.width == m_model_input_shape.height && m_keep_ratio)
        {
            // letterbox
            float col = static_cast<float>(frame.cols);
            float row = static_cast<float>(frame.rows);
            float scale = std::min(m_model_input_shape.width / col, m_model_input_shape.height / row);
            m_scale_factor.x = 1.0f / scale;
            m_scale_factor.y = 1.0f / scale;
            int resized_w = static_cast<int>(col * scale);
            int resized_h = static_cast<int>(row * scale);
            m_input_resized_roi = cv::Rect(0, 0, resized_w, resized_h);
            auto bg_color = mode_color(frame);
            cv::resize(frame, m_resized_frame, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_AREA);
            cv::copyMakeBorder(m_resized_frame,
                               m_resized_frame,
                               0, m_model_input_shape.height - resized_h,
                               0, m_model_input_shape.width - resized_w,
                               cv::BORDER_CONSTANT,
                               bg_color);
        }
        else
        {
            cv::resize(frame, m_resized_frame, m_model_input_shape, 0, 0, cv::INTER_AREA);
            m_scale_factor.x = static_cast<float>(frame.cols) / m_model_input_shape.width;
            m_scale_factor.y = static_cast<float>(frame.rows) / m_model_input_shape.height;
            m_input_resized_roi = cv::Rect(cv::Point{0, 0}, m_model_input_shape);
        }
        if (!m_resized_frame.isContinuous())
            m_resized_frame = m_resized_frame.clone();

        cv::dnn::blobFromImage(m_resized_frame, m_input_blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), true);

        float *input_data = (float *)m_input_blob.data;
        m_input_tensor = ov::Tensor(m_compiled_model.input().get_element_type(), m_compiled_model.input().get_shape(), input_data);
        m_inference_request.set_input_tensor(m_input_tensor);
    }

    inline size_t get_top_idx_from_3(const std::vector<float> &scores)
    {
        size_t top_idx = 0;
        if (scores[1] > scores[top_idx])
            top_idx = 1;
        if (scores[2] > scores[top_idx])
            top_idx = 2;
        return top_idx;
    }
    // Method to postprocess the inference results
    std::vector<Detector::Result> Detector::postProcessing(const cv::Mat &frame, Order order)
    {
        std::vector<size_t> class_list;
        std::vector<float> confidence_list;
        std::vector<cv::Rect> box_list;
        std::vector<int> original_index;

        // Get the output tensor from the inference request

        ov::Tensor dets_tensor = m_inference_request.get_output_tensor(0);
        const float *detections = dets_tensor.data<const float>();
        cv::Mat detection_outputs(m_model_output_shape_det, CV_32F, const_cast<float *>(detections));
        detection_outputs = detection_outputs.clone();

        ov::Tensor segs_tensor = m_inference_request.get_output_tensor(1);
        const float *segments = segs_tensor.data<const float>();
        cv::Mat segment_outputs(m_segment_channel, m_model_output_shape_seg.area(), CV_32F, const_cast<float *>(segments));
        segment_outputs = segment_outputs.clone();
        for (int i = 0; i < m_model_output_shape_det.width; ++i)
        {
            std::vector<float> class_scores{detection_outputs.at<float>(4, i), detection_outputs.at<float>(5, i), detection_outputs.at<float>(6, i)};
            auto class_id = get_top_idx_from_3(class_scores);
            auto score = class_scores.at(class_id);

            // Check if the detection meets the confidence threshold
            if (score > m_confidence_threshold)
            {
                class_list.push_back(class_id);
                confidence_list.push_back(score);
                original_index.push_back(i);

                const float cx = detection_outputs.at<float>(0, i);
                const float cy = detection_outputs.at<float>(1, i);
                const float w = detection_outputs.at<float>(2, i);
                const float h = detection_outputs.at<float>(3, i);

                cv::Rect2d box;
                box.x = cx - w / 2;
                box.y = cy - h / 2;
                box.width = w;
                box.height = h;
                box_list.push_back(box & m_input_resized_roi);
            }
        }

        std::vector<int> NMS_ids;
        cv::dnn::NMSBoxes(box_list, confidence_list, m_confidence_threshold, m_NMS_threshold, NMS_ids);
        std::vector<Result> results;
        if (NMS_ids.empty())
            return results;

        results.reserve(NMS_ids.size());
        std::vector<cv::Mat> mask_coefs;
        mask_coefs.reserve(NMS_ids.size());

        // Collect final detections after NMS
        for (int i = 0; i < NMS_ids.size(); ++i)
        {
            const auto id = NMS_ids[i];
            results.emplace_back(class_list[id], confidence_list[id], scaleBoundingBox(box_list[id]) & cv::Rect(0, 0, frame.cols, frame.rows));
            mask_coefs.emplace_back(detection_outputs.col(original_index[id]).rowRange(m_model_output_shape_det.height - m_segment_channel, m_model_output_shape_det.height).t());
        }
        cv::Mat mask_coef_mat;
        cv::vconcat(mask_coefs, mask_coef_mat);
        cv::Mat masks = mask_coef_mat * segment_outputs;

        float col = static_cast<float>(frame.cols);
        float row = static_cast<float>(frame.rows);
        float scale = std::min(m_model_output_shape_seg.width / col, m_model_output_shape_seg.height / row);
        // m_scale_factor.x = 1.0f / scale;
        // m_scale_factor.y = 1.0f / scale;
        int resized_w = static_cast<int>(col * scale);
        int resized_h = static_cast<int>(row * scale);
        auto roi_seg = cv::Rect(0, 0, resized_w, resized_h);
        for (int i = 0; i < NMS_ids.size(); ++i)
        {
            cv::Mat mask(m_model_output_shape_seg, CV_32FC1, (float *)masks.data + i * m_model_output_shape_seg.area());
            cv::Mat fmask;
            cv::resize(mask(roi_seg), fmask, frame.size(), 0, 0, cv::INTER_CUBIC);
            cv::Mat pure = cv::Mat::zeros(fmask.size(), fmask.type());
            fmask(results[i].box).copyTo(pure(results[i].box));
            cv::compare(pure, 0, results[i].mask, cv::CMP_GT);
        }

        switch (order)
        {
        case Order::AREA_LARGE:
            std::sort(results.begin(), results.end(), [](const Result &a, const Result &b)
                      { return (a.box.area() > b.box.area()); });
            break;
        case Order::AREA_SMALL:
            std::sort(results.begin(), results.end(), [](const Result &a, const Result &b)
                      { return (a.box.area() < b.box.area()); });
            break;
        case Order::X_LEFT:
            std::sort(results.begin(), results.end(), [](const Result &a, const Result &b)
                      { return (a.box.x < b.box.x); });
            break;
        case Order::X_RIGHT:
            std::sort(results.begin(), results.end(), [](const Result &a, const Result &b)
                      { return (a.box.x > b.box.x); });
            break;
        case Order::Y_TOP:
            std::sort(results.begin(), results.end(), [](const Result &a, const Result &b)
                      { return (a.box.y < b.box.y); });
            break;
        case Order::Y_BOTTOM:
            std::sort(results.begin(), results.end(), [](const Result &a, const Result &b)
                      { return (a.box.y > b.box.y); });
            break;

        default:
            break;
        }
        return results;
    }

    void Detector::setupPreprocessing(ov::preprocess::PrePostProcessor &ppp)
    {
        ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255, 255, 255});
        ppp.input().model().set_layout("NCHW");
    }

    void Detector::setupModelOutputShape(std::shared_ptr<const ov::Model> model)
    {
        auto outputs = model->outputs();
        auto output_shape = outputs[0].get_shape();
        m_model_output_shape_det = cv::Size(static_cast<int>(*(output_shape.rbegin())), static_cast<int>(*(output_shape.rbegin() + 1)));

        output_shape = outputs[1].get_shape();
        m_model_output_shape_seg = cv::Size(static_cast<int>(*(output_shape.rbegin())), static_cast<int>(*(output_shape.rbegin() + 1)));
    }

    // Method to get the bounding box in the correct scale
    cv::Rect Detector::scaleBoundingBox(const cv::Rect2d &src) const
    {
        cv::Rect box = src;
        box.x = static_cast<int>(std::round(box.x * m_scale_factor.x));
        box.y = static_cast<int>(std::round(box.y * m_scale_factor.y));
        box.width = static_cast<int>(std::round(box.width * m_scale_factor.x));
        box.height = static_cast<int>(std::round(box.height * m_scale_factor.y));
        return box;
    }
} // namespace yolo
