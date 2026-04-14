#include "trt_infer.hpp"

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <algorithm>
#include <cassert>
#include <map>

#include "cuda_tools.hpp"

class Logger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const char *msg) noexcept override
    {
        if (severity == Severity::kINTERNAL_ERROR)
        {
            LOG_E("trt", "NVInfer INTERNAL_ERROR: %s", msg);
            abort();
        }
        else if (severity == Severity::kERROR)
        {
            LOG_E("trt", "NVInfer: %s", msg);
        }
        else if (severity == Severity::kWARNING)
        {
            LOG_W("trt", "NVInfer: %s", msg);
        }
        else if (severity == Severity::kINFO)
        {
            LOG_I("trt", "NVInfer: %s", msg);
        }
        else
        {
            LOG_D("trt", "NVInfer: %s", msg);
        }
    }
};

static Logger glogger;

namespace trt
{
    // TensorRT object deleter compatible with TRT6/8/10.
    template <typename _T>
    struct NvInferDeleter
    {
        void operator()(_T *ptr) const noexcept
        {
            if (ptr == nullptr)
                return;
#if NV_TENSORRT_MAJOR >= 10
            delete ptr;
#else
            ptr->destroy();
#endif
        }
    };

    template <typename _T>
    static std::shared_ptr<_T> make_nvshared(_T *ptr)
    {
        return std::shared_ptr<_T>(ptr, NvInferDeleter<_T>());
    }

    class EngineContext
    {
    public:
        virtual ~EngineContext() { destroy(); }

        void set_stream(CUStream stream)
        {
            if (owner_stream_)
            {
                if (stream_)
                    cudaStreamDestroy(stream_);
                owner_stream_ = false;
            }
            stream_ = stream;
        }

        bool build_model(const void *pdata, size_t size)
        {
            if (pdata == nullptr || size == 0)
                return false;

            owner_stream_ = true;
            checkCudaRuntime(cudaStreamCreate(&stream_));
            if (stream_ == nullptr)
                return false;

            runtime_ = make_nvshared(nvinfer1::createInferRuntime(glogger));
            if (runtime_ == nullptr)
                return false;

#if NV_TENSORRT_MAJOR >= 10
            engine_ = make_nvshared(runtime_->deserializeCudaEngine(pdata, size));
#else
            engine_ = make_nvshared(runtime_->deserializeCudaEngine(pdata, size, nullptr));
#endif
            if (engine_ == nullptr)
                return false;

            context_ = make_nvshared(engine_->createExecutionContext());
            return context_ != nullptr;
        }

    private:
        void destroy()
        {
            context_.reset();
            engine_.reset();
            runtime_.reset();

            if (owner_stream_)
            {
                if (stream_)
                    cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }
        }

    public:
        cudaStream_t stream_ = nullptr;
        bool owner_stream_ = false;
        std::shared_ptr<nvinfer1::IExecutionContext> context_;
        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        std::shared_ptr<nvinfer1::IRuntime> runtime_;
    };

    class InferImpl : public Infer
    {
    private:
        std::vector<std::shared_ptr<Tensor>> inputs_;
        std::vector<std::shared_ptr<Tensor>> outputs_;
        std::vector<int> inputs_map_to_ordered_index_;
        std::vector<int> outputs_map_to_ordered_index_;
        std::vector<std::string> inputs_name_;
        std::vector<std::string> outputs_name_;
        std::vector<std::shared_ptr<Tensor>> orderdBlobs_;
        std::map<std::string, int> blobsNameMapper_;
        std::shared_ptr<EngineContext> context_;
        std::vector<void *> bindingsPtr_;
        std::shared_ptr<Memory> workspace_;
        int max_batch_size_ = 1;
        int device_ = 0;

    public:
        virtual ~InferImpl();
        virtual bool load(const std::string &file);
        virtual bool load_from_memory(const void *pdata, size_t size);
        virtual void destroy();
        virtual void forward(bool sync) override;
        virtual int get_max_batch_size() const override;
        virtual CUStream get_stream() const override;
        virtual void set_stream(CUStream stream) override;
        virtual void synchronize() const override;
        virtual std::size_t get_device_memory_size() const override;
        virtual std::shared_ptr<Memory> get_workspace() const override;
        virtual std::shared_ptr<Tensor> input(int index = 0) const override;
        virtual std::string get_input_name(int index = 0) const override;
        virtual std::shared_ptr<Tensor> output(int index = 0) const override;
        virtual std::string get_output_name(int index = 0) const override;
        virtual std::shared_ptr<Tensor> tensor(const std::string &name) const override;
        virtual bool is_output_name(const std::string &name) const override;
        virtual bool is_input_name(const std::string &name) const override;
        virtual void set_input(int index, std::shared_ptr<Tensor> tensor) override;
        virtual void set_output(int index, std::shared_ptr<Tensor> tensor) override;
        virtual std::shared_ptr<std::vector<uint8_t>> serial_engine() const override;
        virtual void print() const override;
        virtual int num_output() const override;
        virtual int num_input() const override;
        virtual int device() const override;

    private:
        bool initialize_from_engine_bytes(const void *pdata, size_t size);
        void build_engine_input_and_outputs_mapper();
        bool query_binding_meta(EngineContext *context, int bindingIndex, int maxBatchSize, std::string &bindingName, std::vector<int> &tensorDims, DataType &dtype, bool &isInput);
        void register_binding_tensor(const std::string &bindingName, const std::shared_ptr<Tensor> &tensor, bool isInput, int bindingIndex);
        bool set_input_shapes_by_batch(int inputBatchSize);
        void prepare_output_tensors(int inputBatchSize);
        void refresh_binding_ptrs();
        bool enqueue_inference();
    };

    InferImpl::~InferImpl() { destroy(); }

    void InferImpl::destroy()
    {
        int old_device = 0;
        checkCudaRuntime(cudaGetDevice(&old_device));
        checkCudaRuntime(cudaSetDevice(device_));
        this->context_.reset();
        this->workspace_.reset();
        this->blobsNameMapper_.clear();
        this->outputs_.clear();
        this->inputs_.clear();
        this->inputs_name_.clear();
        this->outputs_name_.clear();
        this->orderdBlobs_.clear();
        this->bindingsPtr_.clear();
        this->inputs_map_to_ordered_index_.clear();
        this->outputs_map_to_ordered_index_.clear();
        checkCudaRuntime(cudaSetDevice(old_device));
    }

    void InferImpl::print() const
    {
        if (!context_)
        {
            LOG_W("trt", "Infer print. nullptr.");
            return;
        }

        LOG_I("trt", "Infer %p detail", this);
        LOG_I("trt", "\tBase device: %s", description().c_str());
        LOG_I("trt", "\tMax Batch Size: %d", this->get_max_batch_size());
        LOG_I("trt", "\tInputs: %d", inputs_.size());
        for (int i = 0; i < inputs_.size(); ++i)
        {
            auto &tensor = inputs_[i];
            auto &name = inputs_name_[i];
            LOG_I("trt", "\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
        }

        LOG_I("trt", "\tOutputs: %d", outputs_.size());
        for (int i = 0; i < outputs_.size(); ++i)
        {
            auto &tensor = outputs_[i];
            auto &name = outputs_name_[i];
            LOG_I("trt", "\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
        }
    }

    std::shared_ptr<std::vector<uint8_t>> InferImpl::serial_engine() const
    {
        auto memory = this->context_->engine_->serialize();
        auto output = std::make_shared<std::vector<uint8_t>>((uint8_t *)memory->data(), (uint8_t *)memory->data() + memory->size());
#if NV_TENSORRT_MAJOR >= 10
        delete memory;
#else
        memory->destroy();
#endif
        return output;
    }

    bool InferImpl::load_from_memory(const void *pdata, size_t size)
    {
        if (pdata == nullptr || size == 0)
            return false;
        return initialize_from_engine_bytes(pdata, size);
    }

    bool InferImpl::load(const std::string &file)
    {
        auto data = utils::fs::load_file(file);
        if (data.empty())
            return false;

        return initialize_from_engine_bytes(data.data(), data.size());
    }

    bool InferImpl::initialize_from_engine_bytes(const void *pdata, size_t size)
    {
        context_.reset(new EngineContext());

        // build model
        if (!context_->build_model(pdata, size))
        {
            context_.reset();
            return false;
        }

        workspace_.reset(new Memory());
        cudaGetDevice(&device_);
        build_engine_input_and_outputs_mapper();
        return true;
    }

    std::size_t InferImpl::get_device_memory_size() const
    {
        EngineContext *context = (EngineContext *)this->context_.get();
        return context->context_->getEngine().getDeviceMemorySize();
    }

    static DataType convert_trt_datatype(nvinfer1::DataType dt)
    {
        switch (dt)
        {
        case nvinfer1::DataType::kFLOAT:
            return DataType::Float32;
        case nvinfer1::DataType::kHALF:
            return DataType::Float16;
        case nvinfer1::DataType::kINT8:
            return DataType::Int8;
        // case nvinfer1::DataType::kINT32: return DataType
        default:
            LOG_E("trt", "Unsupport data type %d", dt);
            return DataType::Float32;
        }
    }

    bool InferImpl::query_binding_meta(
        EngineContext *context,
        int bindingIndex,
        int maxBatchSize,
        std::string &bindingName,
        std::vector<int> &tensorDims,
        DataType &dtype,
        bool &isInput)
    {
#if NV_TENSORRT_MAJOR >= 10
        const char *name = context->engine_->getIOTensorName(bindingIndex);
        if (name == nullptr)
            return false;

        bindingName = name;
        auto dims = context->engine_->getTensorShape(bindingName.c_str());
        auto type = context->engine_->getTensorDataType(bindingName.c_str());
        isInput = context->engine_->getTensorIOMode(bindingName.c_str()) == nvinfer1::TensorIOMode::kINPUT;
#else
        const char *name = context->engine_->getBindingName(bindingIndex);
        if (name == nullptr)
            return false;

        bindingName = name;
        auto dims = context->engine_->getBindingDimensions(bindingIndex);
        auto type = context->engine_->getBindingDataType(bindingIndex);
        isInput = context->engine_->bindingIsInput(bindingIndex);
#endif

        tensorDims.resize(dims.nbDims);
        for (int k = 0; k < dims.nbDims; ++k)
        {
            int d = (int)dims.d[k];
            if (k == 0)
                d = d > 0 ? d : maxBatchSize;
            else if (d <= 0)
                d = 1;
            tensorDims[k] = d;
        }

        dtype = convert_trt_datatype(type);
        return true;
    }

    void InferImpl::register_binding_tensor(
        const std::string &bindingName,
        const std::shared_ptr<Tensor> &tensor,
        bool isInput,
        int bindingIndex)
    {
        if (isInput)
        {
            inputs_.push_back(tensor);
            inputs_name_.push_back(bindingName);
            inputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
        }
        else
        {
            outputs_.push_back(tensor);
            outputs_name_.push_back(bindingName);
            outputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
        }

        blobsNameMapper_[bindingName] = bindingIndex;
        orderdBlobs_.push_back(tensor);
    }

    void InferImpl::build_engine_input_and_outputs_mapper()
    {
        EngineContext *context = (EngineContext *)this->context_.get();
#if NV_TENSORRT_MAJOR >= 10
        int nbBindings = context->engine_->getNbIOTensors();
        int max_batchsize = 1;
        if (context->engine_->getNbOptimizationProfiles() > 0)
        {
            for (int i = 0; i < nbBindings; ++i)
            {
                const char *tensorName = context->engine_->getIOTensorName(i);
                if (tensorName == nullptr)
                    continue;
                if (context->engine_->getTensorIOMode(tensorName) != nvinfer1::TensorIOMode::kINPUT)
                    continue;

                const nvinfer1::Dims max_dims =
                    context->engine_->getProfileShape(tensorName, 0, nvinfer1::OptProfileSelector::kMAX);
                if (max_dims.nbDims > 0 && max_dims.d[0] > 0)
                    max_batchsize = std::max(max_batchsize, static_cast<int>(max_dims.d[0]));
            }
        }
#else
        int nbBindings = context->engine_->getNbBindings();
        int max_batchsize = context->engine_->getMaxBatchSize();
#endif

        inputs_.clear();
        inputs_name_.clear();
        outputs_.clear();
        outputs_name_.clear();
        orderdBlobs_.clear();
        bindingsPtr_.clear();
        blobsNameMapper_.clear();
        inputs_map_to_ordered_index_.clear();
        outputs_map_to_ordered_index_.clear();

        for (int i = 0; i < nbBindings; ++i)
        {
            std::string bindingName;
            std::vector<int> tensor_dims;
            DataType dtype = DataType::Float32;
            bool is_input = false;

            if (!query_binding_meta(context, i, max_batchsize, bindingName, tensor_dims, dtype, is_input))
            {
                LOG_E("trt", "Failed to query binding meta at index %d", i);
                continue;
            }

            auto newTensor = std::make_shared<Tensor>((int)tensor_dims.size(), tensor_dims.data(), dtype);
            newTensor->set_stream(this->context_->stream_);
            newTensor->set_workspace(this->workspace_);

            if (is_input)
            {
                max_batchsize = std::max(max_batchsize, tensor_dims.empty() ? 1 : tensor_dims[0]);
            }

            register_binding_tensor(bindingName, newTensor, is_input, i);
        }
        bindingsPtr_.resize(orderdBlobs_.size());
        max_batch_size_ = std::max(1, max_batchsize);
    }

    bool InferImpl::set_input_shapes_by_batch(int inputBatchSize)
    {
        EngineContext *context = (EngineContext *)context_.get();
#if NV_TENSORRT_MAJOR >= 10
        for (int i = 0; i < context->engine_->getNbIOTensors(); ++i)
        {
            const char *tensorName = context->engine_->getIOTensorName(i);
            auto dims = context->engine_->getTensorShape(tensorName);
            dims.d[0] = inputBatchSize;
            if (context->engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
            {
                if (!context->context_->setInputShape(tensorName, dims))
                    return false;
            }
        }
#else
        for (int i = 0; i < context->engine_->getNbBindings(); ++i)
        {
            auto dims = context->engine_->getBindingDimensions(i);
            dims.d[0] = inputBatchSize;
            if (context->engine_->bindingIsInput(i) && !context->context_->setBindingDimensions(i, dims))
                return false;
        }
#endif
        return true;
    }

    void InferImpl::prepare_output_tensors(int inputBatchSize)
    {
        for (int i = 0; i < (int)outputs_.size(); ++i)
        {
            outputs_[i]->resize_single_dim(0, inputBatchSize);
            outputs_[i]->to_gpu(false);
        }
    }

    void InferImpl::refresh_binding_ptrs()
    {
        for (int i = 0; i < (int)orderdBlobs_.size(); ++i)
            bindingsPtr_[i] = orderdBlobs_[i]->gpu();
    }

    bool InferImpl::enqueue_inference()
    {
        EngineContext *context = (EngineContext *)context_.get();
#if NV_TENSORRT_MAJOR >= 10
        bool execute_result = true;
        for (int i = 0; i < context->engine_->getNbIOTensors(); ++i)
        {
            const char *tensorName = context->engine_->getIOTensorName(i);
            execute_result = execute_result && context->context_->setTensorAddress(tensorName, bindingsPtr_[i]);
        }
        return execute_result && context->context_->enqueueV3(context->stream_);
#else
        void **bingdingPtr = bindingsPtr_.data();
        return context->context_->enqueueV2(bingdingPtr, context->stream_, nullptr);
#endif
    }

    void InferImpl::set_stream(CUStream stream)
    {
        if (!this->context_)
            return;
        this->context_->set_stream(stream);
        for (auto &t : orderdBlobs_)
            t->set_stream(stream);
    }

    CUStream InferImpl::get_stream() const
    {
        if (!this->context_)
            return nullptr;
        return this->context_->stream_;
    }

    int InferImpl::device() const
    {
        return device_;
    }

    void InferImpl::synchronize() const
    {
        if (!context_ || !context_->stream_)
            return;
        checkCudaRuntime(cudaStreamSynchronize(context_->stream_));
    }

    bool InferImpl::is_output_name(const std::string &name) const
    {
        return std::find(outputs_name_.begin(), outputs_name_.end(), name) != outputs_name_.end();
    }

    bool InferImpl::is_input_name(const std::string &name) const
    {
        return std::find(inputs_name_.begin(), inputs_name_.end(), name) != inputs_name_.end();
    }

    void InferImpl::forward(bool sync)
    {
        if (!context_ || inputs_.empty())
        {
            LOG_W("trt", "Infer forward skipped because context or inputs are empty.");
            return;
        }

        int inputBatchSize = inputs_[0]->size(0);
        if (!set_input_shapes_by_batch(inputBatchSize))
        {
            LOG_E("trt", "Failed to set input shapes for batch size %d", inputBatchSize);
            return;
        }

        prepare_output_tensors(inputBatchSize);
        refresh_binding_ptrs();

        bool execute_result = enqueue_inference();
        if (!execute_result)
        {
            auto code = cudaGetLastError();
            LOG_F("trt", "execute fail, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
        }
        if (sync)
            synchronize();
    }

    std::shared_ptr<Memory> InferImpl::get_workspace() const
    {
        return workspace_;
    }

    int InferImpl::num_input() const
    {
        return static_cast<int>(this->inputs_.size());
    }

    int InferImpl::num_output() const
    {
        return static_cast<int>(this->outputs_.size());
    }

    void InferImpl::set_input(int index, std::shared_ptr<Tensor> tensor)
    {
        if (index < 0 || index >= (int)inputs_.size())
        {
            LOG_F("trt", "Input index[%d] out of range [size=%d]", index, inputs_.size());
            return;
        }
        this->inputs_[index] = tensor;
        int order_index = inputs_map_to_ordered_index_[index];
        this->orderdBlobs_[order_index] = tensor;
    }

    void InferImpl::set_output(int index, std::shared_ptr<Tensor> tensor)
    {

        if (index < 0 || index >= (int)outputs_.size())
        {
            LOG_F("trt", "Output index[%d] out of range [size=%d]", index, outputs_.size());
            return;
        }
        this->outputs_[index] = tensor;
        int order_index = outputs_map_to_ordered_index_[index];
        this->orderdBlobs_[order_index] = tensor;
    }

    std::shared_ptr<Tensor> InferImpl::input(int index) const
    {
        if (index < 0 || index >= (int)inputs_.size())
        {
            LOG_F("trt", "Input index[%d] out of range [size=%d]", index, inputs_.size());
            return nullptr;
        }
        return this->inputs_[index];
    }

    std::string InferImpl::get_input_name(int index) const
    {
        if (index < 0 || index >= (int)inputs_.size())
        {
            LOG_F("trt", "Input index[%d] out of range [size=%d]", index, inputs_.size());
            return "";
        }
        return this->inputs_name_[index];
    }

    std::shared_ptr<Tensor> InferImpl::output(int index) const
    {
        if (index < 0 || index >= (int)outputs_.size())
        {
            LOG_F("trt", "Output index[%d] out of range [size=%d]", index, outputs_.size());
            return nullptr;
        }
        return outputs_[index];
    }

    std::string InferImpl::get_output_name(int index) const
    {
        if (index < 0 || index >= (int)outputs_name_.size())
        {
            LOG_F("trt", "Output index[%d] out of range [size=%d]", index, outputs_name_.size());
            return "";
        }
        return outputs_name_[index];
    }

    int InferImpl::get_max_batch_size() const
    {
        assert(this->context_ != nullptr);
#if NV_TENSORRT_MAJOR >= 10
        return max_batch_size_;
#else
        return this->context_->engine_->getMaxBatchSize();
#endif
    }

    std::shared_ptr<Tensor> InferImpl::tensor(const std::string &name) const
    {
        auto node = this->blobsNameMapper_.find(name);
        if (node == this->blobsNameMapper_.end())
        {
            LOG_F("trt", "Could not found the input/output node '%s', please makesure your model", name.c_str());
            return nullptr;
        }
        return orderdBlobs_[node->second];
    }

    std::shared_ptr<Infer> load_infer_from_memory(const void *pdata, size_t size)
    {
        std::shared_ptr<InferImpl> Infer(new InferImpl());
        if (!Infer->load_from_memory(pdata, size))
            Infer.reset();
        return Infer;
    }

    std::shared_ptr<Infer> load_infer(const std::string &file)
    {
        std::shared_ptr<InferImpl> Infer(new InferImpl());
        if (!Infer->load(file))
            Infer.reset();
        return Infer;
    }

    DeviceMemorySummary get_current_device_summary()
    {
        DeviceMemorySummary info;
        checkCudaRuntime(cudaMemGetInfo(&info.available, &info.total));
        return info;
    }

    int get_device_count()
    {
        int count = 0;
        checkCudaRuntime(cudaGetDeviceCount(&count));
        return count;
    }

    int get_device()
    {
        int device = 0;
        checkCudaRuntime(cudaGetDevice(&device));
        return device;
    }

    void set_device(int device_id)
    {
        if (device_id == -1)
            return;
        checkCudaRuntime(cudaSetDevice(device_id));
    }

    bool init_nv_plugins()
    {
        bool ok = initLibNvInferPlugins(&glogger, "");
        if (!ok)
            LOG_E("trt", "init lib nvinfer plugins failed.");
        return ok;
    }
}