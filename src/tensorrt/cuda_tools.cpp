#include "cuda_tools.hpp"

namespace trt
{
    bool __check_driver(CUresult e, const char *call, int line, const char *file)
    {
        if (e != CUDA_SUCCESS)
        {
            const char *message = nullptr;
            const char *name = nullptr;
            cuGetErrorString(e, &message);
            cuGetErrorName(e, &name);
            if (message == nullptr)
                message = "unknown CUDA Driver error";
            if (name == nullptr)
                name = "unknown";
            LOG_E("trt", "CUDA Driver error %s # %s, code = %s [ %d ] in file %s:%d", call, message, name, e, file, line);
            return false;
        }
        return true;
    }

    bool __check_runtime(cudaError_t e, const char *call, int line, const char *file)
    {
        if (e != cudaSuccess)
        {
            LOG_E("trt", "CUDA runtime error %s # %s, code=%s [ %d ] in file %s:%d",
                  call,
                  cudaGetErrorString(e),
                  cudaGetErrorName(e),
                  e, file, line);
            return false;
        }
        return true;
    }

    bool check_device_id(int device_id)
    {
        int device_count = -1;
        if (!checkCudaRuntime(cudaGetDeviceCount(&device_count)))
        {
            LOG_E("trt", "Failed to get device count");
            return false;
        }
        if (device_id < 0 || device_id >= device_count)
        {
            LOG_E("trt", "Invalid device id %d, available device count is %d", device_id, device_count);
            return false;
        }
        return true;
    }

    int current_device_id()
    {
        int device_id = 0;
        checkCudaRuntime(cudaGetDevice(&device_id));
        return device_id;
    }

    std::string description()
    {
        cudaDeviceProp prop;
        size_t free_mem, total_mem;
        int device_id = 0;

        checkCudaRuntime(cudaGetDevice(&device_id));
        checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
        checkCudaRuntime(cudaMemGetInfo(&free_mem, &total_mem));

        return utils::format(
            "[ID %d]<%s>[arch %d.%d][GMEM %.2f GB/%.2f GB]",
            device_id, prop.name, prop.major, prop.minor,
            free_mem / 1024.0 / 1024.0 / 1024.0,
            total_mem / 1024.0 / 1024.0 / 1024.0);
    }

    void display_current_useable_device()
    {
        int device_nums = 0;
        checkCudaRuntime(cudaGetDeviceCount(&device_nums));
        if (device_nums == 0)
            LOG_I("trt", "current no device supporting CUDA");
        else
        {
            int old_device = 0;
            checkCudaRuntime(cudaGetDevice(&old_device));
            LOG_I("trt", "current %d device supporting CUDA", device_nums);

            for (int i = 0; i < device_nums; i++)
            {
                cudaDeviceProp prop;
                size_t free_mem, total_mem;

                checkCudaRuntime(cudaSetDevice(i));
                checkCudaRuntime(cudaGetDeviceProperties(&prop, i));
                checkCudaRuntime(cudaMemGetInfo(&free_mem, &total_mem));

                std::string device_info = utils::format(
                    "<%s>[arch %d.%d][GMEM %.2f GB/%.2f GB]",
                    prop.name, prop.major, prop.minor,
                    free_mem / 1024.0f / 1024.0f / 1024.0f,
                    total_mem / 1024.0f / 1024.0f / 1024.0f);
                LOG_I("trt", "device %d : %s", i, device_info.c_str());
            }
            checkCudaRuntime(cudaSetDevice(old_device));
        }
    }

    bool gpu_used_memory_mb(double &used_mb)
    {
        size_t free_mem = 0;
        size_t total_mem = 0;
        if (!checkCudaRuntime(cudaMemGetInfo(&free_mem, &total_mem)) || total_mem == 0)
            return false;

        used_mb = static_cast<double>(total_mem - free_mem) / (1024.0f * 1024.0f);
        return true;
    }

    AutoDevice::AutoDevice(int device_id)
    {
        if (!checkCudaRuntime(cudaGetDevice(&old)))
        {
            old = -1;
            return;
        }
        if (!check_device_id(device_id))
            return;
    }

    AutoDevice::~AutoDevice()
    {
        if (old > 0)
            checkCudaRuntime(cudaSetDevice(old));
    }

    dim3 grid_dims(int numJobs)
    {
        if (numJobs <= 0)
            return dim3(1);
        int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
        return dim3((numJobs + numBlockThreads - 1) / numBlockThreads);
    }

    dim3 block_dims(int numJobs)
    {
        if (numJobs <= 0)
            return dim3(1);
        return dim3(numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS);
    }

}