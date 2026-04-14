#pragma once

#include <future>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../tensorrt/trt_tensor.hpp"

namespace yoloseg
{
    enum class Type : int
    {
        v8 = 0,
        v11 = 1,
        YOLO26 = 2
    };

    struct InstanceSegmentMap
    {
        int width = 0, height = 0;     // width % 8 == 0
        int left = 0, top = 0;         // 160x160 feature map
        unsigned char *data = nullptr; // is width * height memory

        InstanceSegmentMap(int width, int height);
        virtual ~InstanceSegmentMap();
    };

    struct Box
    {
        float left, top, right, bottom, confidence;
        int class_label;
        std::shared_ptr<InstanceSegmentMap> seg; // mask

        Box() = default;
        Box(float left, float top, float right, float bottom, float confidence, int class_label)
            : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
    };
    typedef std::vector<Box> BoxArray;

    enum class NMSMethod : int
    {
        CPU = 0,    // General, for estimate mAP
        FastGPU = 1 // Fast NMS with a small loss of accuracy in corner cases
    };

    void image_to_tensor(const cv::Mat &image, std::shared_ptr<trt::Tensor> &tensor, int ibatch);
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