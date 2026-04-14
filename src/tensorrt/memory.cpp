#include <string.h>
#include <assert.h>

#include "memory.hpp"
#include "cuda_tools.hpp"

inline static int check_and_trans_device_id(int device_id)
{
    if (device_id != trt::KcurrentDeviceId)
    {
        if (trt::check_device_id(device_id))
            return 0;
        return device_id;
    }
    checkCudaRuntime(cudaGetDevice(&device_id));
    return device_id;
}

namespace trt
{
    Memory::Memory(int device_id)
    {
        device_id_ = check_and_trans_device_id(device_id);
    }

    Memory::Memory(void *cpu, size_t cpu_size, void *gpu, size_t gpu_size, int device_id)
    {
        reference(cpu, cpu_size, gpu, gpu_size, device_id);
    }

    Memory::Memory(Memory &&other) noexcept
    {
        cpu_ = other.cpu_;
        cpu_size_ = other.cpu_size_;
        own_cpu_ = other.own_cpu_;
        gpu_ = other.gpu_;
        gpu_size_ = other.gpu_size_;
        own_gpu_ = other.own_gpu_;
        device_id_ = other.device_id_;

        other.cpu_ = nullptr;
        other.cpu_size_ = 0;
        other.own_cpu_ = true;
        other.gpu_ = nullptr;
        other.gpu_size_ = 0;
        other.own_gpu_ = true;
        other.device_id_ = 0;
    }

    Memory &Memory::operator=(Memory &&other) noexcept
    {
        if (this == &other)
            return *this;
        release();

        cpu_ = other.cpu_;
        cpu_size_ = other.cpu_size_;
        own_cpu_ = other.own_cpu_;
        gpu_ = other.gpu_;
        gpu_size_ = other.gpu_size_;
        own_gpu_ = other.own_gpu_;
        device_id_ = other.device_id_;

        other.cpu_ = nullptr;
        other.cpu_size_ = 0;
        other.own_cpu_ = true;
        other.gpu_ = nullptr;
        other.gpu_size_ = 0;
        other.own_gpu_ = true;
        other.device_id_ = 0;
        return *this;
    }

    void Memory::reference(void *cpu, size_t cpu_size, void *gpu, size_t gpu_size, int device_id)
    {
        release();
        if (cpu == nullptr || cpu_size == 0)
        {
            cpu = nullptr;
            cpu_size = 0;
        }
        if (gpu == nullptr || gpu_size == 0)
        {
            gpu = nullptr;
            gpu_size = 0;
        }
        this->cpu_ = cpu;
        this->cpu_size_ = cpu_size;
        this->gpu_ = gpu;
        this->gpu_size_ = gpu_size;
        this->own_cpu_ = !(cpu && cpu_size > 0);
        this->own_gpu_ = !(gpu && gpu_size > 0);
        device_id_ = check_and_trans_device_id(device_id);
    }

    Memory::~Memory()
    {
        release();
    }

    void Memory::release()
    {
        release_cpu();
        release_gpu();
    }

    void Memory::release_cpu()
    {
        if (cpu_)
        {
            if (own_cpu_)
            {
                AutoDevice auto_e_device(device_id_);
                checkCudaRuntime(cudaFreeHost(cpu_));
            }
            cpu_ = nullptr;
        }
        cpu_size_ = 0;
        own_cpu_ = true;
    }

    void Memory::release_gpu()
    {
        if (gpu_)
        {
            if (own_gpu_)
            {
                AutoDevice auto_e_device(device_id_);
                checkCudaRuntime(cudaFree(gpu_));
            }
            gpu_ = nullptr;
        }
        gpu_size_ = 0;
        own_gpu_ = true;
    }

    void *Memory::cpu(size_t size)
    {
        if (size == 0)
            return cpu_;
        if (cpu_size_ < size)
        {
            release_cpu();
            AutoDevice auto_e_device(device_id_);
            checkCudaRuntime(cudaMallocHost(&cpu_, size));
            assert(cpu_ != nullptr);
            memset(cpu_, 0, size); // 可选：初始化为0，视需求而定
            own_cpu_ = true;
        }
        return cpu_;
    }

    void *Memory::gpu(size_t size)
    {
        if (size == 0)
            return gpu_;
        if (gpu_size_ < size)
        {
            release_gpu();
            AutoDevice auto_e_device(device_id_);
            checkCudaRuntime(cudaMalloc(&gpu_, size));
            checkCudaRuntime(cudaMemset(gpu_, 0, size)); // 可选：初始化为0，视需求而定
            own_gpu_ = true;
        }
        return gpu_;
    }
}