#pragma once

#include <string>
#include <vector>

namespace app_benchmark
{
    // Unified benchmark config supporting all task types (det/obb/seg/pose)
    struct BenchConfig
    {
        std::string config_file;
        std::string engine_file;
        std::string image_dir;
        std::string result_dir = "test";
        std::string yolo_type  = "v8";
        std::string task_type  = "det";      // "det", "obb", "seg", "pose"
        int         save_image_count = 20;
        int         warmup     = 50;
        int         iterations = 500;
        int         batch_size = 1;
        std::string batch_size_expr = "1";
        std::vector<int> batch_sizes{1};
        int         memory_sample_step = 10;
        int         gpu_id     = 0;
    };

    bool load_bench_config(const std::string &config_path, BenchConfig &config);
}
