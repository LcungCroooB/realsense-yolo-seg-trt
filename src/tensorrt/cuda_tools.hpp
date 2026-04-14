#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <string>

#include "../logger/logger_macro.h"
#include "../common/utils.hpp"

// 把cuda/tensorRT高频，易错的底层封装成统一入口

#define GPU_BLOCK_THREADS 512

#define checkCudaDriver(call) trt::__check_driver(call, #call, __LINE__, __FILE__)
#define checkCudaRuntime(call) trt::__check_runtime(call, #call, __LINE__, __FILE__)
#define checkCUDAKernel(...)                                                        \
    __VA_ARGS__;                                                                    \
    do                                                                              \
    {                                                                               \
        cudaError_t cuda_status = cudaPeekAtLastError();                            \
        if (cuda_status != cudaSuccess)                                             \
        {                                                                           \
            LOG_E("trt", "Failed to Launch : %s", cudaGetErrorString(cuda_status)); \
        }                                                                           \
    } while (0);

namespace trt
{
    bool __check_driver(CUresult e, const char *call, int line, const char *file);
    bool __check_runtime(cudaError_t e, const char *call, int line, const char *szfile);
    bool check_device_id(int device_id);

    class AutoDevice // 切换设备
    {
    public:
        AutoDevice(int device_id = 0);
        virtual ~AutoDevice(); // 返回默认

    private:
        int old = -1; // default
    };

    std::string description();             // 拼接当前设备描述字符串（设备名、架构、显存）
    int current_device_id();               // 返回当前线程绑定的 CUDA 设备 id
    void display_current_useable_device(); // 打印可用设备列表
    bool gpu_used_memory_mb(double &used_mb); // 查询当前设备已使用显存（MB）

    // 根据任务量计算1D kernel的grid/block维度
    dim3 grid_dims(int numJobs);
    dim3 block_dims(int numJobs);
}