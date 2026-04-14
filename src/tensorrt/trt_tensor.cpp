#include "trt_tensor.hpp"
#include "cuda_tools.hpp"
#include "../logger/logger_macro.h"

#include <assert.h>
#include <cstring>

namespace trt
{
    int data_type_size(DataType type)
    {
        switch (type)
        {
        case DataType::Float32:
            return sizeof(float);
        case DataType::Float16:
            return sizeof(float16);
        case DataType::Int8:
            return sizeof(int8_t);
        default:
            LOG_E("trt", "Unsupported data type: %d", type);
            return -1;
        }
    }

    const char *data_type_string(DataType type)
    {
        switch (type)
        {
        case DataType::Float32:
            return "Float";
        case DataType::Float16:
            return "Float16";
        case DataType::Int8:
            return "Int8";
        default:
            return "Unknow";
        }
    }

    const char *data_head_string(DataHead head)
    {
        switch (head)
        {
        case DataHead::Init:
            return "Init";
        case DataHead::Device:
            return "Device";
        case DataHead::Host:
            return "Host";
        default:
            return "Unknow";
        }
    }

    inline static int get_device(int deivce_id)
    {
        if (deivce_id != KcurrentDeviceId)
        {
            if (check_device_id(deivce_id))
                return 0;
            return deivce_id;
        }
        checkCudaRuntime(cudaGetDevice(&deivce_id));
        return deivce_id;
    }

    Tensor::Tensor(int ndims, const int *dims, DataType dtype, std::shared_ptr<Memory> data, int device_id)
    {
        this->dtype_ = dtype;
        this->device_id_ = get_device(device_id);
        descriptor_string_[0] = 0;
        setup_data(data);
        resize(ndims, dims);
    }

    Tensor::Tensor(DataType dtype, std::shared_ptr<Memory> data, int device_id)
    {
        this->dtype_ = dtype;
        this->device_id_ = get_device(device_id);
        descriptor_string_[0] = 0;
        shape_string_[0] = 0;
        setup_data(data);
    }

    Tensor::~Tensor()
    {
        release();
    }

    void Tensor::setup_data(std::shared_ptr<Memory> data)
    {
        data_ = data;
        if (data_ == nullptr)
            data_ = std::make_shared<Memory>(device_id_);
        else
            device_id_ = data_->device_id();
        head_ = DataHead::Init;
        if (data_->cpu())
            head_ = DataHead::Host;
        if (data_->gpu())
            head_ = DataHead::Device;
    }

    int Tensor::count(int start_axis) const
    {
        if (start_axis >= 0 && start_axis < (int)shape_.size())
        {
            int size = 1;
            for (int i = start_axis; i < (int)shape_.size(); ++i)
                size *= shape_[i];
            return size;
        }
        else
            return 0;
    }

    int Tensor::numel() const
    {
        int value = shape_.empty() ? 0 : 1;
        for (int i = 0; i < (int)shape_.size(); ++i)
            value *= shape_[i];
        return value;
    }

    Tensor &Tensor::to_cpu(bool copy)
    {
        if (head_ == DataHead::Host)
            return *this;
        if (bytes_ == 0)
        {
            head_ = DataHead::Host;
            return *this;
        }
        head_ = DataHead::Host;
        data_->cpu(bytes_);
        if (copy && data_->gpu() != nullptr)
        {
            AutoDevice auto_e_device(device_id_);
            checkCudaRuntime(cudaMemcpyAsync(data_->cpu(), data_->gpu(), bytes_, cudaMemcpyDeviceToHost, stream_));
            checkCudaRuntime(cudaStreamSynchronize(stream_));
        }
        return *this;
    }

    Tensor &Tensor::to_gpu(bool copy)
    {
        if (head_ == DataHead::Device)
            return *this;
        if (bytes_ == 0)
        {
            head_ = DataHead::Device;
            return *this;
        }
        head_ = DataHead::Device;
        data_->gpu(bytes_);
        if (copy && data_->cpu() != nullptr)
        {
            AutoDevice auto_e_device(device_id_);
            checkCudaRuntime(cudaMemcpyAsync(data_->gpu(), data_->cpu(), bytes_, cudaMemcpyHostToDevice, stream_));
        }
        return *this;
    }

    int Tensor::offset_array(std::size_t size, const int *index_array) const
    {
        assert(size <= shape_.size());
        int value = 0;
        for (int i = 0; i < (int)shape_.size(); ++i)
        {
            if (i < size)
            {
                assert(index_array[i] >= 0 && index_array[i] < shape_[i]);
                value += index_array[i];
            }
            if (i + 1 < (int)shape_.size())
                value *= shape_[i + 1];
        }
        return value;
    }

    int Tensor::offset_array(const std::vector<int> &index) const
    {
        return offset_array(index.size(), index.data());
    }

    Tensor &Tensor::resize(int ndims, const int *dims)
    {
        std::vector<int> setup_dims(ndims);
        for (int i = 0; i < ndims; ++i)
        {
            int dim = dims[i];
            if (dim == -1) // bs == -1
            {
                assert(ndims == shape_.size());
                dim = shape_[i];
            }
            setup_dims[i] = dim;
        }
        this->shape_ = setup_dims;
        this->strides_.resize(shape_.size());

        size_t prev_size = element_size();
        size_t prev_shape = 1;
        for (int i = (int)strides_.size() - 1; i >= 0; --i)
        {
            if (i + 1 < strides_.size())
            {
                prev_size = strides_[i + 1];
                prev_shape = shape_[i + 1];
            }
            strides_[i] = prev_size * prev_shape;
        }
        this->adajust_memory_by_update_dims_or_type();
        this->compute_shape_string();
        return *this;
    }

    Tensor &Tensor::resize(const std::vector<int> &dims)
    {
        return resize(dims.size(), dims.data());
    }

    Tensor &Tensor::resize_single_dim(int idim, int size)
    {
        assert(idim >= 0 && idim < (int)shape_.size());
        auto new_shape = shape_;
        new_shape[idim] = size;
        return resize(new_shape);
    }

    Tensor &Tensor::adajust_memory_by_update_dims_or_type() // 更新需要内存空间
    {
        int needed_size = this->numel() * this->element_size();
        if (needed_size > this->bytes_)
        {
            head_ = DataHead::Init;
        }
        this->bytes_ = needed_size;
        return *this;
    }

    Tensor &Tensor::compute_shape_string() // 打印输出
    {
        shape_string_[0] = 0;
        char *buffer = shape_string_;
        size_t buffer_size = sizeof(shape_string_);
        for (int i = 0; i < (int)shape_.size() && buffer_size > 0; ++i)
        {
            int size = 0;
            if (i < (int)shape_.size() - 1)
                size = snprintf(buffer, buffer_size, "%d x ", shape_[i]);
            else
                size = snprintf(buffer, buffer_size, "%d", shape_[i]);

            if (size < 0)
            {
                shape_string_[0] = 0;
                return *this;
            }

            if ((size_t)size >= buffer_size)
            {
                buffer += buffer_size - 1;
                *buffer = 0;
                break;
            }
            buffer += size;
            buffer_size -= (size_t)size;
        }
        return *this;
    }

    Tensor &Tensor::release()
    {
        data_->release();
        shape_.clear();
        bytes_ = 0;
        head_ = DataHead::Init;

        if (stream_owner_ && stream_ != nullptr)
        {
            AutoDevice auto_e_device(this->device_id_);
            checkCudaRuntime(cudaStreamDestroy(stream_));
        }
        stream_owner_ = false;
        stream_ = nullptr;
        return *this;
    }

    Tensor &Tensor::synchronize()
    {
        AutoDevice auto_e_device(this->device());
        checkCudaRuntime(cudaStreamSynchronize(stream_));
        return *this;
    }

    Tensor &Tensor::copy_from_cpu(size_t offset, const void *src, size_t num_element)
    {
        if (head_ == DataHead::Init)
            to_cpu(false);

        size_t offset_location = offset * element_size();
        if (offset_location >= bytes_)
        {
            LOG_E("trt", "Offset location[%zu] >= bytes_[%zu], out of range", offset_location, bytes_);
            return *this;
        }

        size_t copyed_bytes = num_element * element_size();
        size_t remain_bytes = bytes_ - offset_location;
        if (copyed_bytes > remain_bytes)
        {
            LOG_E("trt", "Copyed bytes[%zu] > remain bytes[%zu], out of range", copyed_bytes, remain_bytes);
            return *this;
        }

        if (head_ == DataHead::Device)
        {
            AutoDevice auto_e_device(this->device());
            checkCudaRuntime(cudaMemcpyAsync((char *)data_->gpu() + offset_location, src, copyed_bytes, cudaMemcpyHostToDevice, stream_));
        }
        else if (head_ == DataHead::Host)
        {
            memcpy((char *)data_->cpu() + offset_location, src, copyed_bytes);
        }
        else
        {
            LOG_E("trt", "Unsupport head type %d", head_);
        }
        return *this;
    }

    Tensor &Tensor::copy_from_gpu(size_t offset, const void *src, size_t num_element, int device_id)
    {
        if (head_ == DataHead::Init)
            to_gpu(false);

        size_t offset_location = offset * element_size();
        if (offset_location >= bytes_)
        {
            LOG_E("trt", "Offset location[%zu] >= bytes_[%zu], out of range", offset_location, bytes_);
            return *this;
        }

        size_t copyed_bytes = num_element * element_size();
        size_t remain_bytes = bytes_ - offset_location;
        if (copyed_bytes > remain_bytes)
        {
            LOG_E("trt", "Copyed bytes[%zu] > remain bytes[%zu], out of range", copyed_bytes, remain_bytes);
            return *this;
        }

        if (head_ == DataHead::Device)
        {
            int current_device_id = get_device(device_id);
            int gpu_device_id = device();
            if (current_device_id != gpu_device_id)
                checkCudaRuntime(cudaMemcpyPeerAsync(gpu_ptr<unsigned char>() + offset_location, gpu_device_id, src, current_device_id, copyed_bytes, stream_));
            else
                checkCudaRuntime(cudaMemcpyAsync(gpu_ptr<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToDevice, stream_));
        }
        else if (head_ == DataHead::Host)
        {
            int current_device_id = get_device(device_id);
            AutoDevice auto_e_device(current_device_id);
            checkCudaRuntime(cudaMemcpyAsync(cpu_ptr<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToHost, stream_));
        }
        else
        {
            LOG_E("trt", "Unsupport head type %d", head_);
        }
        return *this;
    }
}