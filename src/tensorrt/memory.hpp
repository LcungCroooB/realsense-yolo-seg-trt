#pragma once

#include <cstddef>

namespace trt
{
    static const int KcurrentDeviceId = -1; // 当前设备id

    class Memory
    {
    public:
        explicit Memory(int device_id = KcurrentDeviceId);
        Memory(void *cpu, size_t cpu_size, void *gpu, size_t gpu_size, int device_id = KcurrentDeviceId);
        virtual ~Memory();

        inline size_t cpu_size() const { return cpu_size_; }
        inline size_t gpu_size() const { return gpu_size_; }
        inline bool own_cpu() const { return own_cpu_; }
        inline bool own_gpu() const { return own_gpu_; }
        inline int device_id() const { return device_id_; }

        inline void *cpu() const { return cpu_; }
        inline void *gpu() const { return gpu_; }

        void *cpu(size_t size); // 获取cpu内存，size为需要的大小，如果当前内存不足则重新分配
        void *gpu(size_t size); // 获取gpu内存，size为需要的大小，如果当前内存不足则重新分配

        void release_cpu(); // 释放cpu内存
        void release_gpu(); // 释放gpu内存
        void release();     // 释放cpu和gpu内存

        void reference(void *cpu, size_t cpu_size, void *gpu, size_t gpu_size, int device_id = KcurrentDeviceId); // 引用外部内存，不负责释放

        Memory(const Memory &) = delete;            // 禁止拷贝构造
        Memory &operator=(const Memory &) = delete; // 禁止拷贝构造和拷贝赋值
        Memory(Memory &&pther) noexcept;            // 移动构造函数
        Memory &operator=(Memory &&other) noexcept; // 移动赋值运算符

    private:
        void *cpu_ = nullptr; // cpu pinned memory
        size_t cpu_size_ = 0; // 当前可以用size
        bool own_cpu_ = true; // 是否拥有cpu内存，负责释放
        void *gpu_ = nullptr; // gpu memory
        size_t gpu_size_ = 0; // 当前可以用size
        bool own_gpu_ = true; // 是否拥有gpu内存，负责释放
        int device_id_ = 0;   // 绑定的设备id，默认为当前设备
    };

}