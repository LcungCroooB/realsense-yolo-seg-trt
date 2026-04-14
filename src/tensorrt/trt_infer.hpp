#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "trt_tensor.hpp"

namespace trt
{
    class Infer
    {
    public:
        virtual ~Infer() = default;

        virtual void forward(bool sync = true) = 0;
        virtual int get_max_batch_size() const = 0;
        virtual void set_stream(CUStream stream) = 0;
        virtual CUStream get_stream() const = 0;
        virtual void synchronize() const = 0;
        virtual std::size_t get_device_memory_size() const = 0;
        virtual std::shared_ptr<Memory> get_workspace() const = 0;
        virtual std::shared_ptr<Tensor> input(int index = 0) const = 0;
        virtual std::shared_ptr<Tensor> output(int index = 0) const = 0;
        virtual std::shared_ptr<Tensor> tensor(const std::string &name) const = 0;
        virtual std::string get_input_name(int index = 0) const = 0;
        virtual std::string get_output_name(int index = 0) const = 0;
        virtual bool is_output_name(const std::string &name) const = 0;
        virtual bool is_input_name(const std::string &name) const = 0;
        virtual int num_output() const = 0;
        virtual int num_input() const = 0;
        virtual void print() const = 0;
        virtual int device() const = 0;
        virtual void set_input(int index, std::shared_ptr<Tensor> tensor) = 0;
        virtual void set_output(int index, std::shared_ptr<Tensor> tensor) = 0;
        virtual std::shared_ptr<std::vector<uint8_t>> serial_engine() const = 0;
    };

    struct DeviceMemorySummary
    {
        std::size_t total;
        std::size_t available;
    };

    DeviceMemorySummary get_current_device_summary();
    int get_device_count();
    int get_device();

    void set_device(int device_id);
    std::shared_ptr<Infer> load_infer_from_memory(const void *pdata, std::size_t size);
    std::shared_ptr<Infer> load_infer(const std::string &file);
    bool init_nv_plugins();
}