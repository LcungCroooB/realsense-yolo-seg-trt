#pragma once

#include <memory>
#include <vector>
#include <cstdint>

#include "memory.hpp"

// 不强绑定，避免引入cuda.h等头文件，减少编译依赖和时间
struct CUstream_st;
typedef CUstream_st CUstreamRaw;

namespace trt
{
    struct float16
    {
        std::uint16_t bits;
    };

    using CUStream = CUstreamRaw *;

    enum class DataHead : int // 数据驻留位置
    {
        Init = 0,
        Device = 1,
        Host = 2
    };

    enum class DataType : int // Tensor元素类型
    {
        unknow = -1,
        Float32 = 0,
        Float16 = 1,
        Int8 = 2
    };

    const char *data_type_string(DataType type);
    const char *data_head_string(DataHead head);
    int data_type_size(DataType type);

    class Tensor
    {
    public:
        Tensor(const Tensor &other) = delete;
        Tensor &operator=(const Tensor &other) = delete;

        explicit Tensor(int ndims, const int *dims, DataType dtype = DataType::Float32, std::shared_ptr<Memory> data = nullptr, int device_id = KcurrentDeviceId);
        explicit Tensor(DataType dtype = DataType::Float32, std::shared_ptr<Memory> data = nullptr, int device_id = KcurrentDeviceId);
        ~Tensor();

        int count(int start_axis = 0) const;                                                  // return axis 对应size
        int numel() const;                                                                    // return bs*c*h*w
        inline int ndims() const { return shape_.size(); }                                    // return 维度大小
        inline int size(int index) const { return shape_[index]; }                            // return index维度大小
        inline int shape(int index) const { return shape_[index]; }                           // return index维度大小
        inline int batch() const { return shape_[0]; }                                        // return bs
        inline int channel() const { return shape_[1]; }                                      // return c
        inline int height() const { return shape_[2]; }                                       // return h
        inline int width() const { return shape_[3]; }                                        // return w
        inline DataType type() const { return dtype_; }                                       // return F32 F16 IN32 IN8
        inline const std::vector<int> &dims() const { return shape_; }                        // return [bs,c,h,w]
        inline const std::vector<std::size_t> &strides() const { return strides_; }           // return [bs_stride, c_stride, h_stride, w_stride]
        inline int bytes_total() const { return bytes_; }                                     // return 占用字节数
        inline int bytes(int start_axis) const { return count(start_axis) * element_size(); } // return bs/bs*c/bs*c*h/bs*c*h*w * sizeof(DataType)
        inline int element_size() const { return data_type_size(dtype_); }                    // return sizeof(DataType)
        inline DataHead head() const { return head_; }                                        // return 数据驻留位置
        int device() const { return device_id_; }                                             // return 绑定的设备id
        std::shared_ptr<Memory> get_data() const { return data_; }                            // 获取数据内存，返回shared_ptr以共享所有权
        std::shared_ptr<Memory> get_workspace() const { return workspace_; }                  // 获取工作内存，返回shared_ptr以共享所有权
        CUStream get_stream() const { return stream_; }                                       // return 关联的CUDA流

        /** data access**/
        inline void *cpu() const
        {
            const_cast<Tensor *>(this)->to_cpu();
            return data_->cpu();
        }
        inline void *gpu() const
        {
            const_cast<Tensor *>(this)->to_gpu();
            return data_->gpu();
        }

        Tensor &to_cpu(bool copy = true); // 数据加载到cpu
        Tensor &to_gpu(bool copy = true); // 数据加载到gpu

        // 获取数据指针，显示转换为对应数据类型的指针
        template <typename _DT>
        inline const _DT *cpu_ptr() const { return reinterpret_cast<const _DT *>(cpu()); }
        template <typename _DT>
        inline _DT *cpu_ptr() { return reinterpret_cast<_DT *>(cpu()); }
        template <typename _DT>
        inline const _DT *gpu_ptr() const { return reinterpret_cast<const _DT *>(gpu()); }
        template <typename _DT>
        inline _DT *gpu_ptr() { return reinterpret_cast<_DT *>(gpu()); }

        // 支持任意维度的索引，转换为统一的offset_array接口进行计算线性偏移
        template <typename... _Args>
        int offset(int index, _Args... index_args) const
        {
            const int index_array[] = {index, index_args...};
            return offset_array(sizeof...(index_args) + 1, index_array);
        }
        int offset_array(const std::vector<int> &index) const;
        int offset_array(std::size_t size, const int *index_array) const;

        // 获取某个多维索引位置对应的元素指针
        template <typename _DT, typename... _Args>
        inline _DT *cpu_at(int i, _Args &&...args) { return cpu_ptr<_DT>() + offset(i, args...); }
        template <typename _DT, typename... _Args>
        inline _DT *gpu_at(int i, _Args &&...args) { return gpu_ptr<_DT>() + offset(i, args...); }

        // 给 Tensor 绑定 CUDA stream
        Tensor &set_stream(CUStream stream, bool owner = false)
        {
            stream_ = stream;
            stream_owner_ = owner;
            return *this;
        }

        // workspace 是 Tensor 内部使用的临时内存，用户可以通过 set_workspace 提供外部内存以复用，减少内存分配和释放的开销
        Tensor &set_workspace(std::shared_ptr<Memory> workspace)
        {
            workspace_ = workspace;
            return *this;
        }

        template <typename... _Args>
        Tensor &resize(int dim_size, _Args... dim_size_args)
        {
            const int dim_size_array[] = {dim_size, dim_size_args...};
            return resize(sizeof...(dim_size_args) + 1, dim_size_array);
        }
        Tensor &resize(int ndims, const int *dims);
        Tensor &resize(const std::vector<int> &ndims);
        Tensor &resize_single_dim(int idim, int size);
        Tensor &release();
        Tensor &synchronize();
        const char *shape_string() const { return shape_string_; }

        Tensor &copy_from_gpu(std::size_t offset, const void *src, std::size_t num_element, int device_id = KcurrentDeviceId);
        Tensor &copy_from_cpu(std::size_t offset, const void *src, std::size_t num_element);

    private:
        void setup_data(std::shared_ptr<Memory> data);
        Tensor &adajust_memory_by_update_dims_or_type();
        Tensor &compute_shape_string();

    private:
        CUStream stream_ = nullptr;          // 关联的CUDA流
        bool stream_owner_ = false;          // 是否拥有CUDA流，负责释放
        std::shared_ptr<Memory> workspace_;  // 内存管理，负责分配和释放
        std::vector<int> shape_;             // 张量维度信息
        std::vector<std::size_t> strides_;   // 张量步长信息
        std::size_t bytes_ = 0;              // 张量占用的字节数
        DataHead head_ = DataHead::Init;     // 数据驻留位置
        DataType dtype_ = DataType::Float32; // 张量元素类型
        int device_id_ = 0;                  // 绑定的设备id，默认为当前设备
        char shape_string_[100] = {0};       // 维度描述字符串，格式为"bs*c*h*w"，用于日志输出
        char descriptor_string_[100] = {0};  // 张量描述字符串，格式为"float32[bs*c*h*w]"，用于日志输出
        std::shared_ptr<Memory> data_;       // 数据内存，负责分配和释放
    };

}