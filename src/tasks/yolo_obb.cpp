#include "yolo_obb.hpp"

#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "../tensorrt/trt_infer.hpp"
#include "../tensorrt/monopoly_allocator.hpp"
#include "../tensorrt/preprocess_kernel.cuh"
#include "../tensorrt/cuda_tools.hpp"
#include "../tensorrt/template_infer.hpp"
#include "../logger/logger_macro.h"
#include "../common/utils.hpp"

namespace yoloobb
{
    const char *type_name(Type type)
    {
        switch (type)
        {
        case Type::v8:
            return "Yolov8-obb";
        case Type::v11:
            return "Yolov11-obb";
        case Type::YOLO26:
            return "Yolov26-obb";
        default:
            return "Unknown";
        }
    }

    Type type_from_string(const std::string &text)
    {
        if (text == "v11" || text == "V11")
            return Type::v11;
        if (text == "yolo26" || text == "YOLO26")
            return Type::YOLO26;
        return Type::v8;
    }

    void decode_kernel_invoker(
        float *predict, int num_bboxes, int num_classes, float confidence_threshold,
        float *invert_affine_matrix, float *parray,
        int max_objects, cudaStream_t stream, Type type, bool channel_major);
    void nms_kernel_invoker(
        float *parray, float nms_threshold, int max_objects, cudaStream_t stream);

    struct AffineMatrix
    {
        float i2d[6];
        float d2i[6];

        void compute(const cv::Size &from, const cv::Size &to)
        {
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;
            float scale = std::min(scale_x, scale_y);

            i2d[0] = scale;
            i2d[1] = 0;
            i2d[2] = -scale * from.width * 0.5f + to.width * 0.5f + scale * 0.5f - 0.5f;
            i2d[3] = 0;
            i2d[4] = scale;
            i2d[5] = -scale * from.height * 0.5f + to.height * 0.5f + scale * 0.5f - 0.5f;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat()
        {
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    static std::tuple<float, float, float> covariance_matrix(const Box &obb)
    {
        float w = obb.width;
        float h = obb.height;
        float r = obb.angle;
        float a = w * w / 12.0f;
        float b = h * h / 12.0f;

        float cos_r = std::cos(r);
        float sin_r = std::sin(r);

        float a_val = a * cos_r * cos_r + b * sin_r * sin_r;
        float b_val = a * sin_r * sin_r + b * cos_r * cos_r;
        float c_val = (a - b) * sin_r * cos_r;

        return std::make_tuple(a_val, b_val, c_val);
    }

    static float probiou(const Box &obb1, const Box &obb2, float eps = 1e-7f)
    {
        float a1, b1, c1, a2, b2, c2;
        std::tie(a1, b1, c1) = covariance_matrix(obb1);
        std::tie(a2, b2, c2) = covariance_matrix(obb2);

        float x1 = obb1.center_x, y1 = obb1.center_y;
        float x2 = obb2.center_x, y2 = obb2.center_y;

        float den = (a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2) + eps;
        float t1 = ((a1 + a2) * std::pow(y1 - y2, 2) + (b1 + b2) * std::pow(x1 - x2, 2)) / den;
        float t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / den;
        float t3 = std::log(((a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2)) /
                            (4.0f * std::sqrt(std::max(a1 * b1 - c1 * c1, 0.0f)) * std::sqrt(std::max(a2 * b2 - c2 * c2, 0.0f)) + eps) +
                            eps);

        float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
        bd = std::max(std::min(bd, 100.0f), eps);
        float hd = std::sqrt(1.0f - std::exp(-bd) + eps);
        return 1.0f - hd;
    }

    static BoxArray cpu_nms(BoxArray &boxes, float threshold)
    {
        std::sort(boxes.begin(), boxes.end(), [](BoxArray::const_reference a, BoxArray::const_reference b)
                  { return a.confidence > b.confidence; });

        BoxArray output;
        output.reserve(boxes.size());

        std::vector<bool> remove_flags(boxes.size());
        for (int i = 0; i < (int)boxes.size(); ++i)
        {
            if (remove_flags[i])
                continue;

            auto &a = boxes[i];
            output.emplace_back(a);

            for (int j = i + 1; j < (int)boxes.size(); ++j)
            {
                if (remove_flags[j])
                    continue;

                auto &b = boxes[j];
                if (b.class_label == a.class_label && probiou(a, b) >= threshold)
                    remove_flags[j] = true;
            }
        }
        return output;
    }

    using ControllerImpl = InferController<
        cv::Mat,
        BoxArray,
        std::tuple<std::string, int>,
        AffineMatrix>;

    class InferImpl : public Infer, public ControllerImpl
    {
    public:
        virtual ~InferImpl()
        {
            stop();
        }

        bool startup(
            const std::string &file, Type type, int gpuid,
            float confidence_threshold, float nms_threshold,
            NMSMethod nms_method, int max_objects,
            bool use_multi_preprocess_stream)
        {
            normalize_ = trt::Norm::alpha_beta(1.0f / 255.0f, 0.0f, trt::ChannelType::Invert);
            type_ = type;
            use_multi_preprocess_stream_ = use_multi_preprocess_stream;
            confidence_threshold_ = confidence_threshold;
            nms_threshold_ = nms_threshold;
            nms_method_ = nms_method;
            max_objects_ = max_objects;
            return ControllerImpl::startup(std::make_tuple(file, gpuid));
        }

        virtual void worker(std::promise<bool> &result) override
        {
            std::string file = std::get<0>(start_param_);
            int gpuid = std::get<1>(start_param_);

            trt::set_device(gpuid);
            auto engine = trt::load_infer(file);
            if (engine == nullptr)
            {
                LOG_E("yoloobb", "Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();

            const int max_image_bbox = max_objects_;
            const int num_box_element = (type_ == Type::YOLO26) ? 7 : 8;

            trt::Tensor affine_matrix_device(trt::DataType::Float32);
            trt::Tensor output_array_device(trt::DataType::Float32);
            int max_batch_size = engine->get_max_batch_size();
            auto input = engine->tensor("images");
            auto output = engine->is_output_name("output")
                              ? engine->tensor("output")
                              : engine->output(0);
            int num_bboxes = output->size(1);
            int num_classes = output->size(2) - 5;
            bool output_channel_major = false;

            if (type_ == Type::v8 || type_ == Type::v11)
            {
                output_channel_major = output->size(1) < output->size(2);
                if (output_channel_major)
                {
                    num_classes = output->size(1) - 5;
                    num_bboxes = output->size(2);
                }
                else
                {
                    num_classes = output->size(2) - 5;
                    num_bboxes = output->size(1);
                }
            }

            input_width_ = input->size(3);
            input_height_ = input->size(2);
            stream_ = engine->get_stream();
            gpu_ = gpuid;
            tensor_allocator_ = std::make_shared<MonopolyAllocator<trt::Tensor>>(max_batch_size * 2);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            affine_matrix_device.set_stream(stream_);
            affine_matrix_device.resize(max_batch_size, 8).to_gpu();
            output_array_device.set_stream(stream_);
            output_array_device.resize(max_batch_size, 1 + max_image_bbox * num_box_element).to_gpu();

            result.set_value(true);

            std::vector<Job> fetch_jobs;
            while (get_jobs_and_wait(fetch_jobs, max_batch_size))
            {
                int infer_batch_size = (int)fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {
                    auto &job = fetch_jobs[ibatch];
                    auto &mono = job.mono_tensor->data();

                    if (mono->get_stream() != stream_)
                        checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));

                    affine_matrix_device.copy_from_gpu(affine_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);
                output_array_device.to_gpu(false);
                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {
                    auto &job = fetch_jobs[ibatch];
                    (void)job;

                    float *image_based_output = output->gpu_at<float>(ibatch);
                    float *output_array_ptr = output_array_device.gpu_at<float>(ibatch);
                    float *affine_matrix = affine_matrix_device.gpu_at<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                    decode_kernel_invoker(image_based_output, num_bboxes, num_classes, confidence_threshold_, affine_matrix, output_array_ptr, max_image_bbox, stream_, type_, output_channel_major);

                    if (nms_method_ == NMSMethod::FastGPU && type_ != Type::YOLO26)
                        nms_kernel_invoker(output_array_ptr, nms_threshold_, max_image_bbox, stream_);
                }

                output_array_device.to_cpu();
                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {
                    float *parray = output_array_device.cpu_at<float>(ibatch);
                    int count = std::min(max_image_bbox, (int)*parray);
                    auto &job = fetch_jobs[ibatch];
                    auto &image_based_boxes = job.output;

                    for (int i = 0; i < count; ++i)
                    {
                        float *pbox = parray + 1 + i * num_box_element;
                        int label = (int)pbox[6];
                        int keepflag = (type_ == Type::YOLO26) ? 1 : (int)pbox[7];
                        if (keepflag == 1)
                            image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], pbox[5], label);
                    }

                    if (nms_method_ == NMSMethod::CPU && type_ != Type::YOLO26)
                        image_based_boxes = cpu_nms(image_based_boxes, nms_threshold_);
                    job.pro->set_value(image_based_boxes);
                }
                fetch_jobs.clear();
            }

            stream_ = nullptr;
            tensor_allocator_.reset();
            LOG_I("yoloobb", "OBB engine destroyed");
        }

        virtual bool preprocess(Job &job, const cv::Mat &image) override
        {
            if (tensor_allocator_ == nullptr)
            {
                LOG_E("yoloobb", "tensor_allocator_ is nullptr");
                return false;
            }

            if (image.empty())
            {
                LOG_E("yoloobb", "Image is empty");
                return false;
            }

            job.mono_tensor = tensor_allocator_->query();
            if (job.mono_tensor == nullptr)
            {
                LOG_E("yoloobb", "Tensor allocator query failed");
                return false;
            }

            trt::AutoDevice auto_device(gpu_);
            auto &tensor = job.mono_tensor->data();
            trt::CUStream preprocess_stream = nullptr;

            if (tensor == nullptr)
            {
                tensor = std::make_shared<trt::Tensor>();
                tensor->set_workspace(std::make_shared<trt::Memory>());

                if (use_multi_preprocess_stream_)
                {
                    checkCudaRuntime(cudaStreamCreate(&preprocess_stream));
                    tensor->set_stream(preprocess_stream, true);
                }
                else
                {
                    preprocess_stream = stream_;
                    tensor->set_stream(preprocess_stream, false);
                }
            }

            cv::Size input_size(input_width_, input_height_);
            job.additional.compute(image.size(), input_size);

            preprocess_stream = tensor->get_stream();
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image = image.cols * image.rows * 3;
            size_t size_matrix = ::utils::upbound(sizeof(job.additional.d2i), 32);
            auto workspace = tensor->get_workspace();
            uint8_t *gpu_workspace = (uint8_t *)workspace->gpu(size_matrix + size_image);
            float *affine_matrix_device = (float *)gpu_workspace;
            uint8_t *image_device = size_matrix + gpu_workspace;

            uint8_t *cpu_workspace = (uint8_t *)workspace->cpu(size_matrix + size_image);
            float *affine_matrix_host = (float *)cpu_workspace;
            uint8_t *image_host = size_matrix + cpu_workspace;

            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, preprocess_stream));

            trt::warp_affine_bilinear_and_normalize_plane(
                image_device, image.cols * 3, image.cols, image.rows,
                reinterpret_cast<float *>(tensor->gpu()), input_width_, input_height_,
                affine_matrix_device, 114,
                normalize_, preprocess_stream);
            return true;
        }

        virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat> &images) override
        {
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<BoxArray> commit(const cv::Mat &image) override
        {
            return ControllerImpl::commit(image);
        }

    private:
        Type type_ = Type::v8;
        int input_width_ = 0;
        int input_height_ = 0;
        int gpu_ = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_ = 0;
        int max_objects_ = 1024;
        NMSMethod nms_method_ = NMSMethod::FastGPU;
        trt::CUStream stream_ = nullptr;
        bool use_multi_preprocess_stream_ = false;
        trt::Norm normalize_;
    };

    std::shared_ptr<Infer> create_infer(
        const std::string &engine_file, Type type, int gpuid,
        float confidence_threshold, float nms_threshold,
        NMSMethod nms_method, int max_objects,
        bool use_multi_preprocess_stream)
    {
        std::shared_ptr<InferImpl> instance(new InferImpl());
        if (!instance->startup(
                engine_file, type, gpuid, confidence_threshold,
                nms_threshold, nms_method, max_objects, use_multi_preprocess_stream))
            instance.reset();
        return instance;
    }

    void image_to_tensor(const cv::Mat &image, std::shared_ptr<trt::Tensor> &tensor, int ibatch)
    {
        trt::Norm normalize = trt::Norm::alpha_beta(1.0f / 255.0f, 0.0f, trt::ChannelType::Invert);

        cv::Size input_size(tensor->size(3), tensor->size(2));
        AffineMatrix affine;
        affine.compute(image.size(), input_size);

        size_t size_image = image.cols * image.rows * 3;
        size_t size_matrix = ::utils::upbound(sizeof(affine.d2i), 32);
        auto workspace = tensor->get_workspace();
        uint8_t *gpu_workspace = (uint8_t *)workspace->gpu(size_matrix + size_image);
        float *affine_matrix_device = (float *)gpu_workspace;
        uint8_t *image_device = size_matrix + gpu_workspace;

        uint8_t *cpu_workspace = (uint8_t *)workspace->cpu(size_matrix + size_image);
        float *affine_matrix_host = (float *)cpu_workspace;
        uint8_t *image_host = size_matrix + cpu_workspace;
        auto stream = tensor->get_stream();

        memcpy(image_host, image.data, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
        checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream));

        trt::warp_affine_bilinear_and_normalize_plane(
            image_device, image.cols * 3, image.cols, image.rows,
            tensor->gpu_at<float>(ibatch), input_size.width, input_size.height,
            affine_matrix_device, 114,
            normalize, stream);
        tensor->synchronize();
    }

}