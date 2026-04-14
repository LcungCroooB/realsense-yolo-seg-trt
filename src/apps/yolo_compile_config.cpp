#include "yolo_compile_config.hpp"

#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>

#include "yaml_helpers.hpp"
#include "../common/utils.hpp"
#include "../logger/logger_macro.h"

namespace
{
    bool parse_mode(const std::string &precision, trt::Mode &mode)
    {
        if (precision == "fp16" || precision == "FP16") { mode = trt::Mode::FP16; return true; }
        if (precision == "fp32" || precision == "FP32") { mode = trt::Mode::FP32; return true; }
        if (precision == "int8" || precision == "INT8") { mode = trt::Mode::INT8; return true; }
        return false;
    }

    const char *mode_suffix(trt::Mode mode)
    {
        if (mode == trt::Mode::FP32) return "fp32";
        if (mode == trt::Mode::INT8) return "int8";
        return "fp16";
    }
}

namespace app_compile
{
    bool load_compile_tasks(const std::string &config_path, std::vector<CompileTask> &tasks)
    {
        tasks.clear();

        std::string resolved_config;
        std::string config_prefix;
        if (!utils::path::resolve_input_path(config_path, resolved_config, config_prefix))
        {
            LOG_E("app", "Compile config not found: %s", config_path.c_str());
            return false;
        }

        YAML::Node root;
        try
        {
            root = YAML::LoadFile(resolved_config);
        }
        catch (const YAML::Exception &e)
        {
            LOG_E("app", "Failed to parse compile config '%s': %s", resolved_config.c_str(), e.what());
            return false;
        }

        const YAML::Node &global_node      = root["global"];
        const YAML::Node &int8_global_node = root["int8"];

        const int  global_batch_size    = yaml_node::read_int (global_node, "max_batch_size",      1);
        const int  global_workspace_mb  = yaml_node::read_int (global_node, "workspace_mb",      1024);
        const bool global_skip_existing = yaml_node::read_bool(global_node, "skip_existing_engine", true);

        const std::string int8_global_image_dir   = yaml_node::read_str(int8_global_node, "image_directory");
        const std::string int8_global_entropy_file = yaml_node::read_str(int8_global_node, "entropy_calibrator_file");

        const YAML::Node &models = root["models"];
        if (!models || !models.IsSequence())
        {
            LOG_E("app", "Config field 'models' must be a sequence");
            return false;
        }

        for (const YAML::Node &model_node : models)
        {
            const std::string model_name    = yaml_node::read_str(model_node, "name");
            const std::string onnx_file_raw = yaml_node::read_str(model_node, "onnx");
            if (model_name.empty() || onnx_file_raw.empty())
            {
                LOG_E("app", "Each model entry must have 'name' and 'onnx'");
                return false;
            }

            std::string resolved_onnx;
            std::string onnx_prefix;
            if (!utils::path::resolve_input_path(onnx_file_raw, resolved_onnx, onnx_prefix))
            {
                LOG_E("app", "ONNX not found for model %s: %s", model_name.c_str(), onnx_file_raw.c_str());
                return false;
            }

            const std::string yolo_type    = yaml_node::read_str (model_node, "yolo_type",  "v8");
            const std::string task_type    = yaml_node::read_str (model_node, "task_type",  "det");
            const int         batch_size   = yaml_node::read_int (model_node, "max_batch_size", global_batch_size);
            const int         workspace_mb = yaml_node::read_int (model_node, "workspace_mb",        global_workspace_mb);
            const bool        skip_existing = yaml_node::read_bool(model_node, "skip_existing_engine", global_skip_existing);

            const YAML::Node &precision_list = model_node["precisions"];
            if (!precision_list || !precision_list.IsSequence())
            {
                LOG_E("app", "Model %s must define precision list in 'precisions'", model_name.c_str());
                return false;
            }

            const YAML::Node &model_int8_node       = model_node["int8"];
            const std::string int8_image_dir_raw     = yaml_node::read_str(model_int8_node, "image_directory",          int8_global_image_dir);
            const std::string int8_entropy_file_raw  = yaml_node::read_str(model_int8_node, "entropy_calibrator_file",  int8_global_entropy_file);

            for (const YAML::Node &prec_node : precision_list)
            {
                const std::string precision_text = prec_node.as<std::string>("");

                trt::Mode mode = trt::Mode::FP16;
                if (!parse_mode(precision_text, mode))
                {
                    LOG_E("app", "Unsupported precision '%s' in model %s", precision_text.c_str(), model_name.c_str());
                    return false;
                }

                const std::string custom_engine_key = std::string("engine_") + mode_suffix(mode);
                std::string engine_file = yaml_node::read_str(model_node, custom_engine_key);
                if (engine_file.empty())
                    engine_file = "models/" + model_name + "_" + mode_suffix(mode) + ".engine";

                CompileTask task;
                task.model_name          = model_name;
                task.yolo_type           = yolo_type;
                task.task_type           = task_type;
                task.onnx_file           = resolved_onnx;
                task.engine_file         = utils::path::resolve_output_path(
                                               engine_file,
                                               onnx_prefix.empty() ? config_prefix : onnx_prefix);
                task.mode                = mode;
                task.max_batch_size      = static_cast<unsigned int>(batch_size > 0 ? batch_size : 1);
                task.workspace_size      = static_cast<size_t>(workspace_mb > 0 ? workspace_mb : 1024) * 1024ull * 1024ull;
                task.skip_if_engine_exists = skip_existing;

                if (mode == trt::Mode::INT8)
                {
                    if (!int8_image_dir_raw.empty())
                    {
                        std::string resolved_int8_dir;
                        std::string int8_prefix;
                        if (!utils::path::resolve_input_path(int8_image_dir_raw, resolved_int8_dir, int8_prefix))
                        {
                            LOG_E("app", "INT8 image directory not found for model %s: %s",
                                  model_name.c_str(), int8_image_dir_raw.c_str());
                            return false;
                        }
                        task.int8_image_directory = resolved_int8_dir;
                    }

                    if (!int8_entropy_file_raw.empty())
                    {
                        task.int8_entropy_calibrator_file = utils::path::resolve_output_path(
                            int8_entropy_file_raw,
                            onnx_prefix.empty() ? config_prefix : onnx_prefix);
                    }
                }

                tasks.emplace_back(task);
            }
        }

        LOG_I("app", "Loaded %d compile tasks from %s", static_cast<int>(tasks.size()), resolved_config.c_str());
        return !tasks.empty();
    }
}
