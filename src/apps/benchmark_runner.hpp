#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "benchmark_common.hpp"
#include "bench_config.hpp"
#include "../common/utils.hpp"
#include "../logger/logger_macro.h"

namespace app_benchmark
{
    // =========================================================================
    // Generic benchmark runner template
    // Traits classes define task-specific behavior (type, infer creation, drawing)
    // =========================================================================

    template <typename TaskTraits>
    class BenchmarkRunner
    {
    public:
        int run(const BenchConfig &config, int warmup, int iterations)
        {
            if (!utils::fs::exists(config.engine_file))
            {
                LOG_E("bench", "Engine not found: %s", config.engine_file.c_str());
                return -1;
            }

            std::vector<LoadedImage> images;
            if (!load_images_from_dir(config.image_dir, images))
                return -1;

            // Create task-specific infer object
            typename TaskTraits::InferType infer;
            if (!TaskTraits::create_infer(
                    config.engine_file,
                    config.yolo_type,
                    config.gpu_id,
                    infer))
            {
                LOG_E("bench", "Failed to create infer from %s", config.engine_file.c_str());
                return -1;
            }

            LOG_I("bench", "========== %s Benchmark ==========", TaskTraits::task_name);
            LOG_I("bench", "Engine: %s", config.engine_file.c_str());
            LOG_I("bench", "Config: %s", config.config_file.c_str());
            LOG_I("bench", "Image Dir: %s", config.image_dir.c_str());
            LOG_I("bench", "Image Count: %d", static_cast<int>(images.size()));
            LOG_I("bench", "Batch Plan: %s", config.batch_size_expr.c_str());
            LOG_I("bench", "Warmup: %d", warmup);
            LOG_I("bench", "Iterations: %d", iterations);
            LOG_I("bench", "Memory Sample Step: %d", config.memory_sample_step);
            LOG_I("bench", "GPU: %d", config.gpu_id);

            const std::array<int, 5> sweep_batch_set{{1, 2, 4, 8, 16}};
            const bool is_full_sweep_plan = config.batch_sizes.size() == sweep_batch_set.size() &&
                                            std::equal(config.batch_sizes.begin(), config.batch_sizes.end(), sweep_batch_set.begin());

            const std::string report_file = make_summary_report_file(
                config.result_dir,
                TaskTraits::task_name,
                config.engine_file);

            if (!write_summary_header(
                    report_file,
                    TaskTraits::task_name,
                    config.config_file,
                    config.engine_file,
                    TaskTraits::type_name_fn(TaskTraits::get_yolo_type(config.yolo_type)),
                    config.image_dir,
                    static_cast<int>(images.size()),
                    warmup,
                    iterations,
                    config.gpu_id,
                    config.batch_size_expr))
                return -1;

            int previous_batch_size = 0;

            for (size_t run_idx = 0; run_idx < config.batch_sizes.size(); ++run_idx)
            {
                const int batch_size = std::max(1, config.batch_sizes[run_idx]);
                const bool use_batch_commits = (config.batch_sizes.size() > 1) || (batch_size > 1);
                const bool is_sweep_batch = std::find(sweep_batch_set.begin(), sweep_batch_set.end(), batch_size) != sweep_batch_set.end();
                const bool is_sweep_profile = is_full_sweep_plan && is_sweep_batch;

                LOG_I("bench", "------------ Run %d/%d ------------", static_cast<int>(run_idx + 1), static_cast<int>(config.batch_sizes.size()));
                LOG_I("bench", "Batch Size: %d", batch_size);
                LOG_I("bench", "Batch Profile: %s", is_sweep_profile ? "sweep_1_2_4_8_16" : "custom");
                LOG_I("bench", "Infer Path: %s", use_batch_commits ? "commits(vector<cv::Mat>)" : "commit(cv::Mat)");

                std::vector<cv::Mat> batch_images;
                batch_images.reserve(static_cast<size_t>(batch_size));

                auto fill_batch = [&](int iter_index)
                {
                    batch_images.clear();
                    const int start = iter_index * batch_size;
                    for (int j = 0; j < batch_size; ++j)
                    {
                        const int idx = (start + j) % static_cast<int>(images.size());
                        batch_images.emplace_back(images[static_cast<size_t>(idx)].image);
                    }
                };

                std::vector<double> warmup_ms;
                warmup_ms.reserve(static_cast<size_t>(warmup));

                MemorySummary memory_summary;
                double mem_used_mb = 0.0;
                if (query_gpu_used_memory_mb(mem_used_mb))
                {
                    memory_summary.base_used_mb = mem_used_mb;
                    memory_summary.peak_used_mb = mem_used_mb;
                }

                double mem_sum_mb = 0.0;
                int mem_samples = 0;
                int sample_tick = 0;

                auto sample_memory = [&](bool force)
                {
                    ++sample_tick;
                    if (!force && (sample_tick % config.memory_sample_step) != 0)
                        return;

                    if (!query_gpu_used_memory_mb(mem_used_mb))
                        return;

                    memory_summary.peak_used_mb = std::max(memory_summary.peak_used_mb, mem_used_mb);
                    mem_sum_mb += mem_used_mb;
                    ++mem_samples;
                };

                for (int i = 0; i < warmup; ++i)
                {
                    fill_batch(i);
                    const auto t0 = std::chrono::steady_clock::now();
                    if (use_batch_commits)
                        TaskTraits::run_batch_inference(infer, batch_images);
                    else
                        TaskTraits::run_single_inference(infer, batch_images.front());
                    const auto t1 = std::chrono::steady_clock::now();
                    warmup_ms.emplace_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                    sample_memory(false);
                }

                std::vector<double> steady_ms;
                steady_ms.reserve(static_cast<size_t>(iterations));
                const auto bench_start = std::chrono::steady_clock::now();
                for (int i = 0; i < iterations; ++i)
                {
                    fill_batch(i);
                    const auto t0 = std::chrono::steady_clock::now();
                    if (use_batch_commits)
                        TaskTraits::run_batch_inference(infer, batch_images);
                    else
                        TaskTraits::run_single_inference(infer, batch_images.front());
                    const auto t1 = std::chrono::steady_clock::now();
                    steady_ms.emplace_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                    sample_memory(false);
                }
                const auto bench_end = std::chrono::steady_clock::now();
                sample_memory(true);

                if (query_gpu_used_memory_mb(mem_used_mb))
                    memory_summary.end_used_mb = mem_used_mb;
                else
                    memory_summary.end_used_mb = memory_summary.peak_used_mb;

                if (mem_samples > 0)
                    memory_summary.avg_used_mb = mem_sum_mb / static_cast<double>(mem_samples);
                else
                    memory_summary.avg_used_mb = memory_summary.base_used_mb;

                const LatencySummary warmup_summary = summarize_latency(warmup_ms);
                const LatencySummary steady_summary = summarize_latency(steady_ms);

                ProfileSwitchSummary profile_summary;
                if (run_idx > 0 && previous_batch_size != batch_size)
                {
                    profile_summary.applicable = true;
                    profile_summary.from_batch = previous_batch_size;
                    profile_summary.to_batch = batch_size;
                    if (!warmup_ms.empty())
                        profile_summary.first_infer_ms = warmup_ms.front();

                    const int window = std::min(5, static_cast<int>(warmup_ms.size()));
                    if (window > 0)
                    {
                        double sum = 0.0;
                        for (int i = 0; i < window; ++i)
                            sum += warmup_ms[static_cast<size_t>(i)];
                        profile_summary.first_window_avg_ms = sum / static_cast<double>(window);
                    }

                    profile_summary.steady_avg_ms = steady_summary.average_ms;
                    profile_summary.estimated_switch_cost_ms =
                        profile_summary.first_infer_ms - profile_summary.steady_avg_ms;
                }

                const double total_s = std::chrono::duration<double>(bench_end - bench_start).count();
                const double batch_fps = total_s > 0.0 ? static_cast<double>(iterations) / total_s : 0.0;
                const double image_fps = batch_fps * static_cast<double>(batch_size);
                const double jitter_cv = steady_summary.average_ms > 0.0
                                             ? steady_summary.stddev_ms / steady_summary.average_ms
                                             : 0.0;

                LOG_I("bench", "Warmup Avg: %.3f ms/batch", warmup_summary.average_ms);
                LOG_I("bench", "Latency Avg: %.3f ms/batch", steady_summary.average_ms);
                LOG_I("bench", "Latency Min/Max: %.3f / %.3f ms/batch", steady_summary.min_ms, steady_summary.max_ms);
                LOG_I("bench", "Latency P50/P95/P99: %.3f / %.3f / %.3f ms/batch", steady_summary.p50_ms, steady_summary.p95_ms, steady_summary.p99_ms);
                LOG_I("bench", "Latency StdDev: %.3f ms/batch", steady_summary.stddev_ms);
                LOG_I("bench", "Jitter CV: %.4f", jitter_cv);
                LOG_I("bench", "Throughput: %.2f batch/s", batch_fps);
                LOG_I("bench", "Throughput: %.2f image/s", image_fps);
                LOG_I("bench", "Memory Base/Avg/Peak/End: %.1f / %.1f / %.1f / %.1f MB",
                      memory_summary.base_used_mb,
                      memory_summary.avg_used_mb,
                      memory_summary.peak_used_mb,
                      memory_summary.end_used_mb);
                LOG_I("bench", "Memory Peak Delta: %.1f MB", memory_summary.peak_used_mb - memory_summary.base_used_mb);
                if (profile_summary.applicable)
                {
                    LOG_I("bench", "Profile Switch: b%d -> b%d", profile_summary.from_batch, profile_summary.to_batch);
                    LOG_I("bench", "Switch First/WindowAvg/Steady: %.3f / %.3f / %.3f ms",
                          profile_summary.first_infer_ms,
                          profile_summary.first_window_avg_ms,
                          profile_summary.steady_avg_ms);
                    LOG_I("bench", "Estimated Switch Cost: %.3f ms", profile_summary.estimated_switch_cost_ms);
                }
                else
                {
                    LOG_I("bench", "Profile Switch: N/A");
                }
                LOG_I("bench", "====================================");

                if (!append_summary_section(
                        report_file,
                        static_cast<int>(run_idx + 1),
                        static_cast<int>(config.batch_sizes.size()),
                        batch_size,
                        is_sweep_profile,
                        use_batch_commits ? "commits(vector<cv::Mat>)" : "commit(cv::Mat)",
                        warmup_summary,
                        steady_summary,
                        memory_summary,
                        profile_summary,
                        jitter_cv,
                        image_fps,
                        batch_fps))
                    return -1;

                previous_batch_size = batch_size;
            }

            LOG_I("bench", "Benchmark report saved to %s", report_file.c_str());

            // Task-specific visualization
            TaskTraits::save_visualizations(
                config.result_dir,
                config.save_image_count,
                images,
                infer);

            return 0;
        }
    };
}
