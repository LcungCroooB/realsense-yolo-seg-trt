#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../tensorrt/trt_builder.hpp"

namespace app_compile
{
    struct CompileTask
    {
        std::string model_name;
        std::string yolo_type  = "v8";
        std::string task_type  = "det";  // "det", "obb", "seg", "pose"
        std::string onnx_file;
        std::string engine_file;
        trt::Mode mode = trt::Mode::FP16;
        unsigned int max_batch_size = 1;
        size_t workspace_size = 1ul << 30;
        bool skip_if_engine_exists = true;

        std::string int8_image_directory;
        std::string int8_entropy_calibrator_file;
    };

    bool load_compile_tasks(const std::string &config_path, std::vector<CompileTask> &tasks);
}
