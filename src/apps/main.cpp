#include <iostream>
#include <string.h>
#include <string>

#include "platform_info.hpp"
#include "benchmark.hpp"
#include "bench_config.hpp"
#include "d435_seg.hpp"

int app_yolo_compile();

int main(int argc, char **argv)
{
    const char *option = "app";
    const char *task = "compile";

    if (argc < 3)
    {
        std::cerr
            << "usage: ./build/bin/yolo_seg_trt <option> <task> [args...]\n"
            << "\napp tasks:\n"
            << "  ./build/bin/yolo_seg_trt app compile\n"
            << "  ./build/bin/yolo_seg_trt app d435_seg <config_yaml>\n"
            << "\nbench tasks:\n"
            << "  ./build/bin/yolo_seg_trt bench platform_info\n"
            << "  ./build/bin/yolo_seg_trt bench det    <config_yaml>\n"
            << "  ./build/bin/yolo_seg_trt bench obb    <config_yaml>\n"
            << "  ./build/bin/yolo_seg_trt bench seg    <config_yaml>\n"
            << "  ./build/bin/yolo_seg_trt bench pose   <config_yaml>\n"
            << std::endl;
        return -1;
    }

    option = argv[1];
    task = argv[2];

    if (strcmp(option, "app") == 0)
    {
        if (strcmp(task, "compile") == 0)
            return app_yolo_compile();

        if (strcmp(task, "d435_seg") == 0)
        {
            if (argc < 4)
            {
                std::cerr
                    << "usage: ./build/bin/yolo_seg_trt app d435_seg <config_yaml>\n"
                    << "example: ./build/bin/yolo_seg_trt app d435_seg configs/seg_depth.yaml"
                    << std::endl;
                return -1;
            }

            return app_seg::app_d435_seg(argv[3]);
        }

        std::cerr << "Cannot find app task: " << task << std::endl;
        return -1;
    }

    if (strcmp(option, "bench") == 0)
    {
        if (strcmp(task, "platform_info") == 0)
        {
            app_benchmark::print_platform_info();
            return 0;
        }

        if (strcmp(task, "det") == 0 || strcmp(task, "obb") == 0 ||
            strcmp(task, "seg") == 0 || strcmp(task, "pose") == 0)
        {
            if (argc < 4)
            {
                std::cerr
                    << "usage: ./build/bin/yolo_seg_trt bench " << task << " <config_yaml>\n"
                    << "example: ./build/bin/yolo_seg_trt bench " << task << " configs/bench_" << task << ".yaml"
                    << std::endl;
                return -1;
            }

            app_benchmark::BenchConfig config;
            if (!app_benchmark::load_bench_config(argv[3], config))
                return -1;

            if (config.task_type != task)
            {
                std::cerr
                    << "task mismatch: command is '" << task
                    << "' but config task_type is '" << config.task_type << "'\n"
                    << "please run the matching command for this config" << std::endl;
                return -1;
            }

            // Route to appropriate benchmark function
            if (strcmp(task, "det") == 0)
                return app_benchmark::benchmark_det(config);
            else if (strcmp(task, "obb") == 0)
                return app_benchmark::benchmark_obb(config);
            else if (strcmp(task, "seg") == 0)
                return app_benchmark::benchmark_seg(config);
            else if (strcmp(task, "pose") == 0)
                return app_benchmark::benchmark_pose(config);
        }

        std::cerr << "Cannot find benchmark task: " << task << std::endl;
        return -1;
    }

    std::cerr << "Cannot find option: " << option << std::endl;
    return -1;
}