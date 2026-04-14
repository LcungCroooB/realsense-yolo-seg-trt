#pragma once

#include <string>
#include <vector>
#include <functional>
#include "trt_tensor.hpp"

namespace trt
{
    enum class Mode : int
    {
        FP32 = 0,
        FP16 = 1,
        INT8 = 2
    };
    const char *mode_string(Mode type);

    typedef std::function<void(int current, int count, const std::vector<std::string> &files, std::shared_ptr<Tensor> &tensor)> Int8Preprocess;

    struct CompileConfig
    {
        Mode mode = Mode::FP16;        // 模式，默认为FP16
        unsigned int maxBatchSize = 1; // 最大批次大小，默认为1
        std::string source;
        std::string saveto;
        size_t maxWorkspaceSize = 1ul << 30;     // 最大工作空间大小，默认为1GB
        Int8Preprocess int8Preprocess = nullptr; // INT8预处理函数，默认为空
        std::string int8ImageDirectory;          // INT8图像目录，默认为空
        std::string int8EntropyCalibratorFile;   // INT8熵校准文件，默认为空
    };

    bool compile(const CompileConfig &config);
}