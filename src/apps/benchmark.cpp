#include "benchmark.hpp"
#include "benchmark_runner.hpp"
#include "benchmark_traits.hpp"
#include "bench_config.hpp"

namespace app_benchmark
{
    int benchmark_det(const BenchConfig &config)
    {
        BenchmarkRunner<traits::DetTraits> runner;
        return runner.run(config, config.warmup, config.iterations);
    }

    int benchmark_obb(const BenchConfig &config)
    {
        BenchmarkRunner<traits::ObbTraits> runner;
        return runner.run(config, config.warmup, config.iterations);
    }

    int benchmark_seg(const BenchConfig &config)
    {
        BenchmarkRunner<traits::SegTraits> runner;
        return runner.run(config, config.warmup, config.iterations);
    }

    int benchmark_pose(const BenchConfig &config)
    {
        BenchmarkRunner<traits::PoseTraits> runner;
        return runner.run(config, config.warmup, config.iterations);
    }
}
