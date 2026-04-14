#include "yolo_compile_config.hpp"

#include "../common/utils.hpp"
#include "../logger/logger_macro.h"
#include "../tasks/yolo.hpp"
#include "../tasks/yolo_obb.hpp"
#include "../tasks/yolo_seg.hpp"
#include "../tasks/yolo_pose.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace
{
    trt::Int8Preprocess build_int8_preprocess(const std::string &task_type, const std::string &yolo_type_str)
    {
        if (task_type == "obb")
        {
            return [](int current, int count, const std::vector<std::string> &files, std::shared_ptr<trt::Tensor> &tensor)
            {
                LOG_I("app", "INT8 preprocess (obb) %d / %d", current, count);
                for (size_t i = 0; i < files.size(); ++i)
                {
                    cv::Mat image = cv::imread(files[i]);
                    if (image.empty()) { LOG_W("app", "INT8 image is empty: %s", files[i].c_str()); continue; }
                    yoloobb::image_to_tensor(image, tensor, static_cast<int>(i));
                }
            };
        }
        if (task_type == "seg")
        {
            return [](int current, int count, const std::vector<std::string> &files, std::shared_ptr<trt::Tensor> &tensor)
            {
                LOG_I("app", "INT8 preprocess (seg) %d / %d", current, count);
                for (size_t i = 0; i < files.size(); ++i)
                {
                    cv::Mat image = cv::imread(files[i]);
                    if (image.empty()) { LOG_W("app", "INT8 image is empty: %s", files[i].c_str()); continue; }
                    yoloseg::image_to_tensor(image, tensor, static_cast<int>(i));
                }
            };
        }
        if (task_type == "pose")
        {
            return [](int current, int count, const std::vector<std::string> &files, std::shared_ptr<trt::Tensor> &tensor)
            {
                LOG_I("app", "INT8 preprocess (pose) %d / %d", current, count);
                for (size_t i = 0; i < files.size(); ++i)
                {
                    cv::Mat image = cv::imread(files[i]);
                    if (image.empty()) { LOG_W("app", "INT8 image is empty: %s", files[i].c_str()); continue; }
                    yolopose::image_to_tensor(image, tensor, static_cast<int>(i));
                }
            };
        }
        // default: det
        const yolo::Type det_type = yolo::type_from_string(yolo_type_str);
        return [det_type](int current, int count, const std::vector<std::string> &files, std::shared_ptr<trt::Tensor> &tensor)
        {
            LOG_I("app", "INT8 preprocess (det) %d / %d", current, count);
            for (size_t i = 0; i < files.size(); ++i)
            {
                cv::Mat image = cv::imread(files[i]);
                if (image.empty()) { LOG_W("app", "INT8 image is empty: %s", files[i].c_str()); continue; }
                yolo::image_to_tensor(image, tensor, static_cast<int>(i), det_type);
            }
        };
    }

    bool compile_one_model(const app_compile::CompileTask &task)
    {
        if (!utils::fs::exists(task.onnx_file))
        {
            LOG_E("app", "ONNX not found: %s", task.onnx_file.c_str());
            return false;
        }

        if (task.skip_if_engine_exists && utils::fs::exists(task.engine_file))
        {
            LOG_I("app", "Skip compile, engine already exists: %s", task.engine_file.c_str());
            return true;
        }

        trt::CompileConfig cfg;
        cfg.mode = task.mode;
        cfg.maxBatchSize = task.max_batch_size;
        cfg.source = task.onnx_file;
        cfg.saveto = task.engine_file;
        cfg.maxWorkspaceSize = task.workspace_size;

        if (cfg.mode == trt::Mode::INT8)
        {
            cfg.int8ImageDirectory = task.int8_image_directory;
            cfg.int8EntropyCalibratorFile = task.int8_entropy_calibrator_file;
            cfg.int8Preprocess = build_int8_preprocess(task.task_type, task.yolo_type);
        }

        LOG_I(
            "app",
            "Compile model[%s]: %s -> %s (mode=%s, max_batch=%u)",
            task.model_name.c_str(),
            cfg.source.c_str(),
            cfg.saveto.c_str(),
            trt::mode_string(cfg.mode),
            cfg.maxBatchSize
        );

        const bool ok = trt::compile(cfg);
        if (!ok)
            LOG_E("app", "Compile failed: %s", cfg.source.c_str());
        return ok;
    }
}

int app_yolo_compile()
{
    std::vector<app_compile::CompileTask> tasks;
    if (!app_compile::load_compile_tasks("configs/yolo_compile.yaml", tasks))
    {
        LOG_E("app", "Load compile tasks failed");
        return -1;
    }

    bool all_ok = true;
    for (const app_compile::CompileTask &task : tasks)
        all_ok = compile_one_model(task) && all_ok;

    LOG_I("app", "Compile finished, status=%s", all_ok ? "success" : "failed");
    return all_ok ? 0 : -1;
}
