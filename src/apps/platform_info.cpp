#include "platform_info.hpp"

#include <NvInferVersion.h>
#include <opencv2/core/version.hpp>
#include <sys/utsname.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include "../logger/logger_macro.h"
#include "../tensorrt/cuda_tools.hpp"
#include "../tensorrt/trt_infer.hpp"

namespace
{
    std::string trim(const std::string &text)
    {
        const char *spaces = " \t\r\n";
        const size_t begin = text.find_first_not_of(spaces);
        if (begin == std::string::npos)
            return std::string();
        const size_t end = text.find_last_not_of(spaces);
        return text.substr(begin, end - begin + 1);
    }

    std::string read_first_cpu_model_name()
    {
        std::ifstream input("/proc/cpuinfo");
        std::string line;
        while (std::getline(input, line))
        {
            const std::string key = "model name";
            const size_t pos = line.find(key);
            if (pos == std::string::npos)
                continue;
            const size_t colon = line.find(':', pos + key.size());
            if (colon == std::string::npos)
                continue;
            return trim(line.substr(colon + 1));
        }
        return "unknown";
    }

    std::string read_mem_total()
    {
        std::ifstream input("/proc/meminfo");
        std::string line;
        while (std::getline(input, line))
        {
            const std::string key = "MemTotal:";
            if (line.find(key) != 0)
                continue;

            std::istringstream iss(line.substr(key.size()));
            long long mem_kb = 0;
            std::string unit;
            iss >> mem_kb >> unit;
            if (mem_kb <= 0)
                return "unknown";

            const double mem_gb = static_cast<double>(mem_kb) / 1024.0 / 1024.0;
            char buffer[64] = {0};
            std::snprintf(buffer, sizeof(buffer), "%.2f GB", mem_gb);
            return buffer;
        }
        return "unknown";
    }

    std::string hostname()
    {
        char buffer[256] = {0};
        if (gethostname(buffer, sizeof(buffer) - 1) != 0)
            return "unknown";
        buffer[sizeof(buffer) - 1] = '\0';
        return buffer;
    }

    std::string kernel_release()
    {
        struct utsname info;
        if (uname(&info) != 0)
            return "unknown";
        return info.release;
    }

    std::string cuda_version_string(int version)
    {
        if (version <= 0)
            return "unknown";
        const int major = version / 1000;
        const int minor = (version % 1000) / 10;
        char buffer[32] = {0};
        std::snprintf(buffer, sizeof(buffer), "%d.%d", major, minor);
        return buffer;
    }
}

namespace app_benchmark
{
    void print_platform_info()
    {
        LOG_I("bench", "================ Platform Info ================");
        LOG_I("bench", "Host: %s", hostname().c_str());
        LOG_I("bench", "Kernel: %s", kernel_release().c_str());
        LOG_I("bench", "CPU: %s", read_first_cpu_model_name().c_str());
        LOG_I("bench", "Logical CPUs: %u", std::thread::hardware_concurrency());
        LOG_I("bench", "Memory: %s", read_mem_total().c_str());

        LOG_I("bench", "CUDA Toolkit: %s", cuda_version_string(CUDART_VERSION).c_str());

        const int current_device = trt::get_device();
        const int device_count = trt::get_device_count();
        LOG_I("bench", "GPU Count: %d", device_count);
        if (device_count > 0)
        {
            for (int device_id = 0; device_id < device_count; ++device_id)
            {
                trt::set_device(device_id);
                LOG_I("bench", "GPU[%d]: %s", device_id, trt::description().c_str());
            }
            trt::set_device(current_device);

            const trt::DeviceMemorySummary memory = trt::get_current_device_summary();
            LOG_I(
                "bench",
                "Current Device Memory: available=%.2f GB, total=%.2f GB",
                static_cast<double>(memory.available) / 1024.0 / 1024.0 / 1024.0,
                static_cast<double>(memory.total) / 1024.0 / 1024.0 / 1024.0
            );
        }
        else
        {
            LOG_W("bench", "GPU Count: unavailable");
        }

        LOG_I("bench", "TensorRT: %d.%d.%d", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
        LOG_I("bench", "OpenCV: %s", CV_VERSION);
        LOG_I("bench", "Compiler: %s", __VERSION__);
        LOG_I("bench", "================================================");
    }
}