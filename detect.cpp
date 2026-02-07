#include "detect.h"
#include <memory>
#include <iostream>
#include <vector>
#include "auxi_funcs.hpp"

namespace yolo
{
    int Detector::InitVino(const std::string &model_path)
    {
        std::shared_ptr<ov::Model> model;
        try
        {
            model = ov_core.read_model(model_path);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Caught exception: " << e.what() << std::endl;
            return UNKNOWN_ERROR;
        }

        // Get input shape from the model
        const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
        const ov::PartialShape input_shape = inputs[0].get_partial_shape();
        if (input_shape[2].is_static() && input_shape[3].is_static())
        {
            m_model_input_shape = cv::Size(static_cast<int>(input_shape[3].get_length()), static_cast<int>(input_shape[2].get_length()));
        }

        // Support dynamic shapes only for RECT
        if (m_preprocessing_method == PreprocessingMethod::RECT)
        {
            model->reshape({1, 3, ov::Dimension(), ov::Dimension()});
        }
        else
        {
            model->reshape({1, 3, static_cast<size_t>(m_model_input_shape.height), static_cast<size_t>(m_model_input_shape.width)});
        }

        try
        {
            m_compiled_model = ov_core.compile_model(model, "CPU");
            m_inference_request = m_compiled_model.create_infer_request();
        }
        catch (const ov::Exception &e)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Caught ov::Exception : " << e.what() << std::endl;
            return UNKNOWN_ERROR;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Caught const std::exception : " << e.what() << std::endl;
            return UNKNOWN_ERROR;
        }
        catch (...)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Caught Unknown exception " << std::endl;
            return UNKNOWN_ERROR;
        }

        setupModelOutputShape(model);
        return SUCCESS;
    }

    Detector::Detector(float confidence_threshold, float NMS_threshold, int imgsz, PreprocessingMethod method)
        : Detector(confidence_threshold, NMS_threshold, cv::Size(imgsz, imgsz), method)
    {
    }
    Detector::Detector(float confidence_threshold, float NMS_threshold, cv::Size input_shape, PreprocessingMethod method)
    {
        m_confidence_threshold = confidence_threshold;
        m_NMS_threshold = NMS_threshold;
        m_model_input_shape = input_shape;
        m_preprocessing_method = method;
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
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Caught ov::Exception : " << e.what() << std::endl;
            return StatusCode::INFERENCE_ERROR;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Caught std::exception : " << e.what() << std::endl;
            return StatusCode::INFERENCE_ERROR;
        }
        catch (...)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Caught Unknown exception " << std::endl;
            return StatusCode::INFERENCE_ERROR;
        }
        results = postProcessing(frame, order);
        return StatusCode::SUCCESS;
    }

    // Method to preprocess the input frame
    void Detector::preProcessing(const cv::Mat &frame)
    {
        int target_w = m_model_input_shape.width;
        int target_h = m_model_input_shape.height;
        m_pad = cv::Point2i(0, 0);

        if (m_preprocessing_method == PreprocessingMethod::RECT)
        {
            float scale = std::min(static_cast<float>(target_w) / frame.cols, static_cast<float>(target_h) / frame.rows);
            int resized_w = static_cast<int>(std::round(frame.cols * scale));
            int resized_h = static_cast<int>(std::round(frame.rows * scale));

            target_w = (resized_w + m_stride - 1) / m_stride * m_stride;
            target_h = (resized_h + m_stride - 1) / m_stride * m_stride;

            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_AREA);

            m_pad.x = (target_w - resized_w) / 2;
            m_pad.y = (target_h - resized_h) / 2;

            auto bg_color = mode_color(frame);
            m_resized_frame = cv::Mat(cv::Size(target_w, target_h), frame.type(), bg_color);
            resized.copyTo(m_resized_frame(cv::Rect(m_pad.x, m_pad.y, resized_w, resized_h)));

            m_scale_factor.x = static_cast<float>(frame.cols) / resized_w;
            m_scale_factor.y = static_cast<float>(frame.rows) / resized_h;
            m_input_resized_roi = cv::Rect(m_pad.x, m_pad.y, resized_w, resized_h);
        }
        else if (m_preprocessing_method == PreprocessingMethod::LETTERBOX)
        {
            float scale = std::min(static_cast<float>(target_w) / frame.cols, static_cast<float>(target_h) / frame.rows);
            int resized_w = static_cast<int>(std::round(frame.cols * scale));
            int resized_h = static_cast<int>(std::round(frame.rows * scale));

            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_AREA);

            m_pad.x = (target_w - resized_w) / 2;
            m_pad.y = (target_h - resized_h) / 2;

            auto bg_color = mode_color(frame);
            m_resized_frame = cv::Mat(cv::Size(target_w, target_h), frame.type(), bg_color);
            resized.copyTo(m_resized_frame(cv::Rect(m_pad.x, m_pad.y, resized_w, resized_h)));

            m_scale_factor.x = 1.0f / scale;
            m_scale_factor.y = 1.0f / scale;
            m_input_resized_roi = cv::Rect(m_pad.x, m_pad.y, resized_w, resized_h);
        }
        else // RESIZE
        {
            cv::resize(frame, m_resized_frame, cv::Size(target_w, target_h), 0, 0, cv::INTER_AREA);
            m_scale_factor.x = static_cast<float>(frame.cols) / target_w;
            m_scale_factor.y = static_cast<float>(frame.rows) / target_h;
            m_input_resized_roi = cv::Rect(0, 0, target_w, target_h);
        }

        if (!m_resized_frame.isContinuous())
            m_resized_frame = m_resized_frame.clone();

        cv::dnn::blobFromImage(m_resized_frame, m_input_blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), true);

        float *input_data = (float *)m_input_blob.data;
        auto blob_size = m_input_blob.size;
        ov::Shape input_shape = {static_cast<size_t>(blob_size[0]), static_cast<size_t>(blob_size[1]), static_cast<size_t>(blob_size[2]), static_cast<size_t>(blob_size[3])};
        m_input_tensor = ov::Tensor(m_compiled_model.input().get_element_type(), input_shape, input_data);
        m_inference_request.set_input_tensor(m_input_tensor);
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
        auto dets_shape = dets_tensor.get_shape();
        m_model_output_shape_det = cv::Size(static_cast<int>(dets_shape[2]), static_cast<int>(dets_shape[1]));
        const float *detections = dets_tensor.data<const float>();
        cv::Mat detection_outputs(m_model_output_shape_det, CV_32F, const_cast<float *>(detections));
        detection_outputs = detection_outputs.clone();

        int classes_num = m_model_output_shape_det.height - 4;
        bool has_seg = false;
        cv::Mat segment_outputs;
        try {
            ov::Tensor segs_tensor = m_inference_request.get_output_tensor(1);
            auto segs_shape = segs_tensor.get_shape();
            m_model_output_shape_seg = cv::Size(static_cast<int>(segs_shape[3]), static_cast<int>(segs_shape[2]));
            m_segment_channel = static_cast<int>(segs_shape[1]);
            const float *segments = segs_tensor.data<const float>();
            segment_outputs = cv::Mat(m_segment_channel, m_model_output_shape_seg.area(), CV_32F, const_cast<float *>(segments));
            segment_outputs = segment_outputs.clone();
            classes_num -= m_segment_channel;
            has_seg = true;
        } catch (...) {
            has_seg = false;
        }

        for (int i = 0; i < m_model_output_shape_det.width; ++i)
        {
            float score = 0;
            int class_id = -1;
            for (int j = 0; j < classes_num; ++j) {
                float s = detection_outputs.at<float>(4 + j, i);
                if (s > score) {
                    score = s;
                    class_id = j;
                }
            }

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
            if (has_seg)
                mask_coefs.emplace_back(detection_outputs.col(original_index[id]).rowRange(m_model_output_shape_det.height - m_segment_channel, m_model_output_shape_det.height).t());
        }

        if (has_seg && !mask_coefs.empty()) {
            cv::Mat mask_coef_mat;
            cv::vconcat(mask_coefs, mask_coef_mat);
            cv::Mat masks = mask_coef_mat * segment_outputs;

            float seg_scale_x = static_cast<float>(m_model_output_shape_seg.width) / m_resized_frame.cols;
            float seg_scale_y = static_cast<float>(m_model_output_shape_seg.height) / m_resized_frame.rows;
            auto roi_seg = cv::Rect(
                static_cast<int>(m_input_resized_roi.x * seg_scale_x),
                static_cast<int>(m_input_resized_roi.y * seg_scale_y),
                static_cast<int>(m_input_resized_roi.width * seg_scale_x),
                static_cast<int>(m_input_resized_roi.height * seg_scale_y));
            for (int i = 0; i < NMS_ids.size(); ++i)
            {
                cv::Mat mask(m_model_output_shape_seg, CV_32FC1, (float *)masks.data + i * m_model_output_shape_seg.area());
                cv::Mat fmask;
                cv::resize(mask(roi_seg), fmask, frame.size(), 0, 0, cv::INTER_CUBIC);
                cv::Mat pure = cv::Mat::zeros(fmask.size(), fmask.type());
                fmask(results[i].box).copyTo(pure(results[i].box));
                cv::compare(pure, 0, results[i].mask, cv::CMP_GT);
            }
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


    void Detector::setupModelOutputShape(std::shared_ptr<const ov::Model> model)
    {
        auto outputs = model->outputs();
        auto output_shape0 = outputs[0].get_partial_shape();
        int det_h = output_shape0[1].is_static() ? static_cast<int>(output_shape0[1].get_length()) : 0;
        int det_w = output_shape0[2].is_static() ? static_cast<int>(output_shape0[2].get_length()) : 0;
        m_model_output_shape_det = cv::Size(det_w, det_h);

        if (outputs.size() > 1)
        {
            auto output_shape1 = outputs[1].get_partial_shape();
            if (output_shape1[1].is_static())
                m_segment_channel = static_cast<int>(output_shape1[1].get_length());
            int seg_h = output_shape1[2].is_static() ? static_cast<int>(output_shape1[2].get_length()) : 0;
            int seg_w = output_shape1[3].is_static() ? static_cast<int>(output_shape1[3].get_length()) : 0;
            m_model_output_shape_seg = cv::Size(seg_w, seg_h);
        }
    }

    // Method to get the bounding box in the correct scale
    cv::Rect Detector::scaleBoundingBox(const cv::Rect2d &src) const
    {
        cv::Rect box;
        box.x = static_cast<int>(std::round((src.x - m_pad.x) * m_scale_factor.x));
        box.y = static_cast<int>(std::round((src.y - m_pad.y) * m_scale_factor.y));
        box.width = static_cast<int>(std::round(src.width * m_scale_factor.x));
        box.height = static_cast<int>(std::round(src.height * m_scale_factor.y));
        return box;
    }
} // namespace yolo
