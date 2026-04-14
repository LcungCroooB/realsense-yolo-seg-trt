#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


#include "../common/utils.hpp"
#include "../logger/logger_macro.h"
#include "../tensorrt/cuda_tools.hpp"

namespace app_benchmark
{
    // =========================================================================
    // Common structures and utilities for all task benchmarks (det/obb/seg/pose)
    // =========================================================================

    struct LatencySummary
    {
        double average_ms = 0.0;
        double min_ms = 0.0;
        double max_ms = 0.0;
        double p50_ms = 0.0;
        double p95_ms = 0.0;
        double p99_ms = 0.0;
        double stddev_ms = 0.0;
    };

    struct LoadedImage
    {
        std::string path;
        cv::Mat image;
    };

    struct MemorySummary
    {
        double base_used_mb = 0.0;
        double avg_used_mb = 0.0;
        double peak_used_mb = 0.0;
        double end_used_mb = 0.0;
    };

    struct ProfileSwitchSummary
    {
        bool applicable = false;
        int from_batch = 0;
        int to_batch = 0;
        double first_infer_ms = 0.0;
        double first_window_avg_ms = 0.0;
        double steady_avg_ms = 0.0;
        double estimated_switch_cost_ms = 0.0;
    };

    inline bool query_gpu_used_memory_mb(double &used_mb)
    {
        return trt::gpu_used_memory_mb(used_mb);
    }

    // -- Latency helpers --

    inline double percentile_sorted(const std::vector<double> &sorted_values, double p)
    {
        if (sorted_values.empty())
            return 0.0;
        if (sorted_values.size() == 1)
            return sorted_values.front();

        const double rank = p * static_cast<double>(sorted_values.size() - 1);
        const size_t low = static_cast<size_t>(std::floor(rank));
        const size_t high = static_cast<size_t>(std::ceil(rank));
        const double weight = rank - static_cast<double>(low);
        return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight;
    }

    inline LatencySummary summarize_latency(const std::vector<double> &latencies_ms)
    {
        LatencySummary output;
        if (latencies_ms.empty())
            return output;

        output.min_ms = *std::min_element(latencies_ms.begin(), latencies_ms.end());
        output.max_ms = *std::max_element(latencies_ms.begin(), latencies_ms.end());

        const double sum = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
        output.average_ms = sum / static_cast<double>(latencies_ms.size());

        std::vector<double> sorted = latencies_ms;
        std::sort(sorted.begin(), sorted.end());
        output.p50_ms = percentile_sorted(sorted, 0.50);
        output.p95_ms = percentile_sorted(sorted, 0.95);
        output.p99_ms = percentile_sorted(sorted, 0.99);

        double variance = 0.0;
        for (double x : latencies_ms)
        {
            const double d = x - output.average_ms;
            variance += d * d;
        }
        variance /= static_cast<double>(latencies_ms.size());
        output.stddev_ms = std::sqrt(variance);
        return output;
    }

    // -- Image loading helpers --

    inline bool load_images_from_dir(const std::string &image_dir, std::vector<LoadedImage> &images)
    {
        const std::vector<std::string> files =
            utils::fs::find_files(image_dir, "*.jpg;*.jpeg;*.png;*.bmp", false, true);
        if (files.empty())
        {
            LOG_E("bench", "No images found in %s", image_dir.c_str());
            return false;
        }

        images.clear();
        images.reserve(files.size());
        for (const std::string &file : files)
        {
            cv::Mat image = cv::imread(file);
            if (image.empty())
                continue;
            LoadedImage item;
            item.path = file;
            item.image = image;
            images.emplace_back(std::move(item));
        }

        if (images.empty())
        {
            LOG_E("bench", "All images are empty under %s", image_dir.c_str());
            return false;
        }
        return true;
    }

    // -- Utility functions --

    inline int clamp_to_int(float value, int lower, int upper)
    {
        return std::max(lower, std::min(static_cast<int>(std::round(value)), upper));
    }

    // -- Report writing helpers --

    inline std::string make_summary_report_file(
        const std::string &result_dir,
        const std::string &task_name,
        const std::string &engine_file)
    {
        const std::string engine_name = utils::fs::file_name(engine_file, false);
        return utils::format(
            "%s/%s_model_%s_%lld.log",
            result_dir.c_str(),
            task_name.c_str(),
            engine_name.c_str(),
            static_cast<long long>(utils::time::timestamp_millisecond()));
    }

    inline bool write_summary_header(
        const std::string &report_file,
        const std::string &task_name,
        const std::string &config_file,
        const std::string &engine_file,
        const std::string &yolo_type_name,
        const std::string &image_dir,
        int image_count,
        int warmup,
        int iterations,
        int gpu_id,
        const std::string &batch_plan)
    {
        FILE *fp = utils::fs::fopen_mkdirs(report_file, "wb");
        if (fp == nullptr)
        {
            LOG_E("bench", "Failed to create benchmark report: %s", report_file.c_str());
            return false;
        }

        const std::string title = task_name == "DET"
                                      ? "Single Model Benchmark"
                                      : task_name + " Model Benchmark";

        std::string content;
        content += "========== ";
        content += title;
        content += " ==========";
        content += "\nConfig File: " + config_file;
        content += "\nEngine: " + engine_file;
        content += "\nYolo Type: " + yolo_type_name;
        content += "\nImage Dir: " + image_dir;
        content += "\nImage Count: " + std::to_string(image_count);
        content += "\nBatch Plan: " + batch_plan;
        content += "\nWarmup: " + std::to_string(warmup);
        content += "\nIterations: " + std::to_string(iterations);
        content += "\nGPU: " + std::to_string(gpu_id);
        content += "\n";
        for (size_t i = 0; i < title.size() + 20; ++i)
            content += "=";
        content += "\n";

        fwrite(content.data(), 1, content.size(), fp);
        fclose(fp);
        return true;
    }

    inline bool append_summary_section(
        const std::string &report_file,
        int run_index,
        int run_count,
        int batch_size,
        bool is_sweep_batch,
        const char *infer_path,
        const LatencySummary &warmup_summary,
        const LatencySummary &steady_summary,
        const MemorySummary &memory_summary,
        const ProfileSwitchSummary &profile_summary,
        double jitter_cv,
        double image_fps,
        double batch_fps)
    {
        FILE *fp = std::fopen(report_file.c_str(), "ab");
        if (fp == nullptr)
        {
            LOG_E("bench", "Failed to append benchmark report: %s", report_file.c_str());
            return false;
        }

        std::string content;
        content += "\n------------ Run ";
        content += std::to_string(run_index);
        content += "/";
        content += std::to_string(run_count);
        content += " ------------";
        content += "\nBatch Size: " + std::to_string(batch_size);
        content += "\nBatch Profile: ";
        content += is_sweep_batch ? "sweep_1_2_4_8_16" : "custom";
        content += "\nInfer Path: ";
        content += infer_path;
        content += "\nWarmup Avg: " + utils::format("%.3f ms/batch", warmup_summary.average_ms);
        content += "\nLatency Avg: " + utils::format("%.3f ms/batch", steady_summary.average_ms);
        content += "\nLatency Min/Max: " + utils::format("%.3f / %.3f ms/batch", steady_summary.min_ms, steady_summary.max_ms);
        content += "\nLatency P50/P95/P99: " + utils::format("%.3f / %.3f / %.3f ms/batch", steady_summary.p50_ms, steady_summary.p95_ms, steady_summary.p99_ms);
        content += "\nLatency StdDev: " + utils::format("%.3f ms/batch", steady_summary.stddev_ms);
        content += "\nJitter CV: " + utils::format("%.4f", jitter_cv);
        content += "\nThroughput: " + utils::format("%.2f batch/s", batch_fps);
        content += "\nThroughput: " + utils::format("%.2f image/s", image_fps);
        content += "\nMemory Base/Avg/Peak/End: " + utils::format(
                                                        "%.1f / %.1f / %.1f / %.1f MB",
                                                        memory_summary.base_used_mb,
                                                        memory_summary.avg_used_mb,
                                                        memory_summary.peak_used_mb,
                                                        memory_summary.end_used_mb);
        content += "\nMemory Peak Delta: " + utils::format(
                                                 "%.1f MB",
                                                 memory_summary.peak_used_mb - memory_summary.base_used_mb);
        if (profile_summary.applicable)
        {
            content += "\nProfile Switch: " + utils::format(
                                                  "b%d -> b%d",
                                                  profile_summary.from_batch,
                                                  profile_summary.to_batch);
            content += "\nSwitch First Infer: " + utils::format("%.3f ms", profile_summary.first_infer_ms);
            content += "\nSwitch First-Window Avg: " + utils::format("%.3f ms", profile_summary.first_window_avg_ms);
            content += "\nSwitch Steady Avg: " + utils::format("%.3f ms", profile_summary.steady_avg_ms);
            content += "\nEstimated Switch Cost: " + utils::format("%.3f ms", profile_summary.estimated_switch_cost_ms);
        }
        else
        {
            content += "\nProfile Switch: N/A";
        }
        content += "\n";

        fwrite(content.data(), 1, content.size(), fp);
        fclose(fp);
        return true;
    }

    template <typename T_Type>
    inline bool write_summary_report(
        const std::string &result_dir,
        const std::string &engine_file,
        const std::string &config_file,
        const char *task_name,
        T_Type yolo_type,
        const char *(*type_name_fn)(T_Type),
        const std::string &image_dir,
        int image_count,
        int batch_size,
        bool is_sweep_batch,
        int warmup,
        int iterations,
        int gpu_id,
        const LatencySummary &warmup_summary,
        const LatencySummary &steady_summary,
        double jitter_cv,
        double image_fps,
        double batch_fps)
    {
        const std::string report_file = make_summary_report_file(result_dir, task_name, engine_file);
        if (!write_summary_header(
                report_file,
                task_name,
                config_file,
                engine_file,
                type_name_fn(yolo_type),
                image_dir,
                image_count,
                warmup,
                iterations,
                gpu_id,
                std::to_string(batch_size)))
            return false;

        if (!append_summary_section(
                report_file,
                1,
                1,
                batch_size,
                is_sweep_batch,
                batch_size > 1 ? "commits(vector<cv::Mat>)" : "commit(cv::Mat)",
                warmup_summary,
                steady_summary,
                MemorySummary{},
                ProfileSwitchSummary{},
                jitter_cv,
                image_fps,
                batch_fps))
            return false;

        LOG_I("bench", "Benchmark report saved to %s", report_file.c_str());
        return true;
    }
}
