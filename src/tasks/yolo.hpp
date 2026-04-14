#pragma once

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

#include "../tensorrt/trt_tensor.hpp"

namespace yolo
{
    enum class Type : int
    {
        v5 = 0,
        X = 1,
        v8 = 2,
        v11 = 3,
        YOLO26 = 4
    };

    struct Box
    {
        float left, top, right, bottom, confidence;
        int class_label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int class_label)
            : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
    };

    typedef std::vector<Box> BoxArray;

    enum class NMSMethod : int
    {
        CPU = 0,    // generic CPU implementation, for esitmate mAP
        FastGPU = 1 // Fast NMS with a small loss of accuracy in corner cases
    };

    void image_to_tensor(const cv::Mat &image, std::shared_ptr<trt::Tensor> &tensor, int ibatch, Type type = Type::v8);
    const char *type_name(Type type);
    Type type_from_string(const std::string &text);

    class Infer
    {
    public:
        virtual std::shared_future<BoxArray> commit(const cv::Mat &image) = 0;
        virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat> &images) = 0;
    };

    std::shared_ptr<Infer> create_infer(
        const std::string &engine_file, Type type, int gpuid,
        float confidence_threshold = 0.5f, float nms_threshold = 0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false);

}