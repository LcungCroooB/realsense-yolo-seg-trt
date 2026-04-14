#include "bench_config.hpp"

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "yaml_helpers.hpp"
#include "../common/utils.hpp"
#include "../logger/logger_macro.h"

namespace
{
    std::string trim(const std::string &text)
    {
        size_t left = 0;
        size_t right = text.size();
        while (left < right && std::isspace(static_cast<unsigned char>(text[left])))
            ++left;
        while (right > left && std::isspace(static_cast<unsigned char>(text[right - 1])))
            --right;
        return text.substr(left, right - left);
    }

    std::vector<int> parse_batch_size_expr(const std::string &expr)
    {
        std::vector<int> output;
        std::string normalized = expr;
        std::replace(normalized.begin(), normalized.end(), ',', '/');

        size_t begin = 0;
        while (begin <= normalized.size())
        {
            size_t end = normalized.find('/', begin);
            if (end == std::string::npos)
                end = normalized.size();

            const std::string token = trim(normalized.substr(begin, end - begin));
            if (!token.empty())
            {
                try
                {
                    const int value = std::stoi(token);
                    if (value > 0)
                        output.emplace_back(value);
                }
                catch (...)
                {
                }
            }

            if (end == normalized.size())
                break;
            begin = end + 1;
        }

        if (output.empty())
            output.emplace_back(1);
        return output;
    }
}

namespace app_benchmark
{
    bool load_bench_config(const std::string &config_path, BenchConfig &config)
    {
        std::string resolved;
        std::string prefix;
        if (!utils::path::resolve_input_path(config_path, resolved, prefix))
        {
            LOG_E("bench", "Benchmark config not found: %s", config_path.c_str());
            return false;
        }

        YAML::Node root;
        try
        {
            root = YAML::LoadFile(resolved);
        }
        catch (const YAML::Exception &e)
        {
            LOG_E("bench", "Failed to parse benchmark config '%s': %s", resolved.c_str(), e.what());
            return false;
        }

        const std::string engine_file_raw = yaml_node::read_str(root, "engine_file");
        if (engine_file_raw.empty())
        {
            LOG_E("bench", "Config '%s' must define 'engine_file'", resolved.c_str());
            return false;
        }

        const std::string image_dir_raw = yaml_node::read_str(root, "image_dir");
        if (image_dir_raw.empty())
        {
            LOG_E("bench", "Config '%s' must define 'image_dir'", resolved.c_str());
            return false;
        }

        // Resolve engine path (must exist)
        std::string resolved_engine;
        std::string engine_prefix;
        if (!utils::path::resolve_input_path(engine_file_raw, resolved_engine, engine_prefix))
        {
            LOG_E("bench", "Engine not found: %s", engine_file_raw.c_str());
            return false;
        }

        // Resolve image dir (must exist)
        std::string resolved_image_dir;
        std::string image_dir_prefix;
        if (!utils::path::resolve_input_path(image_dir_raw, resolved_image_dir, image_dir_prefix))
        {
            LOG_E("bench", "Image dir not found: %s", image_dir_raw.c_str());
            return false;
        }

        config.engine_file = resolved_engine;
        config.image_dir   = resolved_image_dir;
        config.config_file = resolved;
        config.result_dir  = utils::path::resolve_output_path(yaml_node::read_str(root, "result_dir", "test"), prefix);
        config.save_image_count = yaml_node::read_int(root, "save_image_count", 20);
        config.yolo_type   = yaml_node::read_str(root, "yolo_type",  "v8");
        config.task_type   = yaml_node::read_str(root, "task_type",  "det");
        config.warmup      = yaml_node::read_int(root, "warmup",      50);
        config.iterations  = yaml_node::read_int(root, "iterations", 500);
        config.memory_sample_step = yaml_node::read_int(root, "memory_sample_step", 10);
        config.gpu_id      = yaml_node::read_int(root, "gpu_id",       0);

        YAML::Node batch_node = root["batch_size"];
        if (!batch_node)
            config.batch_size_expr = "1";
        else
            config.batch_size_expr = trim(batch_node.as<std::string>());

        config.batch_sizes = parse_batch_size_expr(config.batch_size_expr);
        config.batch_size = config.batch_sizes.front();

        if (config.batch_size <= 0)
            config.batch_size = 1;
        if (config.memory_sample_step <= 0)
            config.memory_sample_step = 1;

        LOG_I("bench", "Loaded bench config from %s (task_type=%s, batch_size=%s, runs=%d, memory_sample_step=%d)",
              resolved.c_str(),
              config.task_type.c_str(),
              config.batch_size_expr.c_str(),
              static_cast<int>(config.batch_sizes.size()),
              config.memory_sample_step);
        return true;
    }
}


