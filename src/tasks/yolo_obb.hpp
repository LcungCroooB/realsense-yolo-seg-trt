#pragma once 

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

#include "../tensorrt/trt_tensor.hpp"

namespace yoloobb
{
    enum class Type: int 
    {
        v8 = 0,
        v11 = 1,
        YOLO26 = 2
    };

    struct Box{
        float center_x, center_y, width, height, angle, confidence;
        int class_label;

        Box() = default;
        Box(float cx, float cy, float w, float h, float a, float conf, int cls_id)
        : center_x(cx), center_y(cy), width(w), height(h), angle(a), confidence(conf), class_label(cls_id) {}
    };
    typedef std::vector<Box> BoxArray;

    enum class NMSMethod: int
    {
        CPU = 0,
        FastGPU = 1
    };

    void image_to_tensor(const cv::Mat& image, std::shared_ptr<trt::Tensor> &tensor, int ibatch);
    const char*type_name(Type type);
    Type type_from_string(const std::string &text);

    class Infer
    {
    public:
        virtual std::shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat>& images) = 0;
    };

    std::shared_ptr<Infer> create_infer(
        const std::string& engine_file, Type type, int gpuid,
        float confidence_threshold = 0.5f, float nms_threshold = 0.5f, 
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );
}
