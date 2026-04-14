#pragma once

#include "bench_config.hpp"
#include <string>

namespace app_benchmark
{
    // Task-specific benchmarks (unified new API)
    int benchmark_det(const BenchConfig &config);
    int benchmark_obb(const BenchConfig &config);
    int benchmark_seg(const BenchConfig &config);
    int benchmark_pose(const BenchConfig &config);
}
