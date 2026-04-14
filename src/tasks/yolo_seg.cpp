#include "yolo_seg.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <queue>

#include "../tensorrt/trt_infer.hpp"
#include "../tensorrt/monopoly_allocator.hpp"
#include "../tensorrt/preprocess_kernel.cuh"
#include "../tensorrt/cuda_tools.hpp"
#include "../tensorrt/template_infer.hpp"
#include "../logger/logger_macro.h"
#include "../common/utils.hpp"

namespace yoloseg
{
	const char *type_name(Type type)
	{
		switch (type)
		{
		case Type::v8:
			return "Yolov8-Seg";
		case Type::v11:
			return "Yolov11-Seg";
		case Type::YOLO26:
			return "Yolov26-Seg";
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

	void affine_project(float *matrix, float x, float y, float *ox, float *oy);

	void decode_kernel_invoker(
		float *predict, int num_bboxes, int num_classes, float confidence_threshold,
		float *invert_affine_matrix, float *parray,
		int max_objects, cudaStream_t stream, Type type, bool channel_major);
	void launch_gather_mask_weights(
		float *predict, int num_bboxes, int base_channel, int row_index, int mask_dim, float *weights_out, cudaStream_t stream);

	void nms_kernel_invoker(
		float *parray, float nms_threshold, int max_objects, cudaStream_t stream);

	void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
							int mask_width, int mask_height, unsigned char *mask_out,
							int mask_dim, int out_width, int out_height, cudaStream_t stream);

	InstanceSegmentMap::InstanceSegmentMap(int width, int height)
	{
		this->width = width;
		this->height = height;
		checkCudaRuntime(cudaMallocHost(&this->data, width * height));
	}

	InstanceSegmentMap::~InstanceSegmentMap()
	{
		if (this->data)
		{
			checkCudaRuntime(cudaFreeHost(this->data));
			this->data = nullptr;
		}
		this->width = 0;
		this->height = 0;
	}

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
	};

	static float iou(const Box &a, const Box &b)
	{
		float cleft = std::max(a.left, b.left);
		float ctop = std::max(a.top, b.top);
		float cright = std::min(a.right, b.right);
		float cbottom = std::min(a.bottom, b.bottom);

		float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
		if (c_area == 0.0f)
			return 0.0f;

		float a_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top);
		float b_area = std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top);
		return c_area / (a_area + b_area - c_area);
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
				if (b.class_label == a.class_label)
				{
					if (iou(a, b) >= threshold)
						remove_flags[j] = true;
				}
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
				LOG_E("yoloseg", "Engine %s load failed", file.c_str());
				result.set_value(false);
				return;
			}

			engine->print();

			const int max_image_bbox = max_objects_;
			const int num_box_element = (type_ == Type::YOLO26) ? 7 : 8;
			trt::Tensor affine_matrix_device(trt::DataType::Float32);
			trt::Tensor output_boxarray_device(trt::DataType::Float32);
			trt::Tensor box_mask_output_memory(trt::DataType::Int8);
			trt::Tensor mask_weights_device(trt::DataType::Float32);
			int max_batch_size = engine->get_max_batch_size();
			auto input = engine->tensor("images");
			auto bbox_head_output = engine->tensor("output0");
			auto mask_head_output = engine->tensor("output1");
			int num_bboxes = bbox_head_output->size(1);
			int num_classes = bbox_head_output->size(2) - 4 - mask_head_output->size(1);
			bool output_channel_major = false;

			if (type_ == Type::v8 || type_ == Type::v11)
			{
				output_channel_major = bbox_head_output->size(1) < bbox_head_output->size(2);
				if (output_channel_major)
				{
					num_bboxes = bbox_head_output->size(2);
					num_classes = bbox_head_output->size(1) - 4 - mask_head_output->size(1);
				}
				else
				{
					num_bboxes = bbox_head_output->size(1);
					num_classes = bbox_head_output->size(2) - 4 - mask_head_output->size(1);
				}
			}

			input_width_ = input->size(3);
			input_height_ = input->size(2);
			tensor_allocator_ = std::make_shared<MonopolyAllocator<trt::Tensor>>(max_batch_size * 2);
			stream_ = engine->get_stream();
			gpu_ = gpuid;
			result.set_value(true);

			input->resize_single_dim(0, max_batch_size).to_gpu();
			affine_matrix_device.set_stream(stream_);
			affine_matrix_device.resize(max_batch_size, 8).to_gpu();

			// Align scratch bytes to 32-byte boundary before bbox array payload.
			output_boxarray_device.resize(max_batch_size, 1 + 31 + max_image_bbox * num_box_element).to_gpu();
			output_boxarray_device.set_stream(stream_);
			mask_weights_device.set_stream(stream_);
			mask_weights_device.resize(mask_head_output->size(1)).to_gpu();

			std::vector<Job> fetch_jobs;
			int batch_log_count = 0;
			while (get_jobs_and_wait(fetch_jobs, max_batch_size))
			{
				int infer_batch_size = (int)fetch_jobs.size();
				if (batch_log_count < 5)
				{
					LOG_I("yoloseg", "runtime infer_batch_size=%d, engine_max_batch=%d", infer_batch_size, max_batch_size);
					++batch_log_count;
				}
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
				output_boxarray_device.to_gpu(false);
				for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
				{
					float *image_based_output = bbox_head_output->gpu_at<float>(ibatch);
					float *output_array_ptr = output_boxarray_device.gpu_at<float>(ibatch);
					float *affine_matrix = affine_matrix_device.gpu_at<float>(ibatch);

					checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
					decode_kernel_invoker(image_based_output, num_bboxes, num_classes, confidence_threshold_, affine_matrix, output_array_ptr, max_image_bbox, stream_, type_, output_channel_major);

					if (nms_method_ == NMSMethod::FastGPU && type_ != Type::YOLO26)
						nms_kernel_invoker(output_array_ptr, nms_threshold_, max_image_bbox, stream_);
				}

				output_boxarray_device.to_cpu();
				for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
				{
					float *parray = output_boxarray_device.cpu_at<float>(ibatch);
					int count = std::min(max_image_bbox, (int)*parray);
					auto &job = fetch_jobs[ibatch];
					auto &image_based_boxes = job.output;
					for (int i = 0; i < count; ++i)
					{
						float *pbox = parray + 1 + i * num_box_element;
						int keepflag = (type_ == Type::YOLO26) ? 1 : (int)pbox[6];
						if (keepflag == 1)
						{
							Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], (int)pbox[5]);

							int row_index = (type_ == Type::YOLO26) ? (int)pbox[6] : (int)pbox[7];
							int mask_dim = mask_head_output->size(1);

							float *mask_weights = nullptr;
							float *mask_head_predict = mask_head_output->gpu_at<float>(ibatch);

							if (output_channel_major && (type_ == Type::v8 || type_ == Type::v11))
							{
								float *mask_weights_device_ptr = mask_weights_device.gpu_ptr<float>();
								launch_gather_mask_weights(
									bbox_head_output->gpu_at<float>(ibatch),
									num_bboxes,
									4 + num_classes,
									row_index,
									mask_dim,
									mask_weights_device_ptr,
									stream_);
								mask_weights = mask_weights_device_ptr;
							}
							else
							{
								mask_weights = bbox_head_output->gpu_at<float>(ibatch) + row_index * bbox_head_output->size(2) + num_classes + 4;
							}

							float left, top, right, bottom;
							float *i2d = job.additional.i2d;
							affine_project(i2d, pbox[0], pbox[1], &left, &top);
							affine_project(i2d, pbox[2], pbox[3], &right, &bottom);

							float box_width = right - left;
							float box_height = bottom - top;
							float scale_to_predict_x = mask_head_output->size(3) / (float)input_width_;
							float scale_to_predict_y = mask_head_output->size(2) / (float)input_height_;
							int mask_out_width = (int)(box_width * scale_to_predict_x + 0.5f);
							int mask_out_height = (int)(box_height * scale_to_predict_y + 0.5f);

							if (mask_out_width > 0 && mask_out_height > 0)
							{
								int bytes_of_mask_out = mask_out_width * mask_out_height;
								box_mask_output_memory.resize(bytes_of_mask_out).to_gpu();
								box_mask_output_memory.to_gpu(false);

								result_object_box.seg = std::make_shared<InstanceSegmentMap>(mask_out_width, mask_out_height);
								unsigned char *mask_out_device = reinterpret_cast<unsigned char *>(box_mask_output_memory.gpu());
								unsigned char *mask_out_host = result_object_box.seg->data;

								decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y, mask_weights,
												   mask_head_predict, mask_head_output->size(3), mask_head_output->size(2),
												   mask_out_device, mask_dim, mask_out_width, mask_out_height, stream_);

								result_object_box.seg->left = (int)(left * scale_to_predict_x);
								result_object_box.seg->top = (int)(top * scale_to_predict_y);
								checkCudaRuntime(cudaMemcpyAsync(mask_out_host, mask_out_device, bytes_of_mask_out, cudaMemcpyDeviceToHost, stream_));
								image_based_boxes.emplace_back(std::move(result_object_box));
							}
						}
					}

					if (nms_method_ == NMSMethod::CPU && type_ != Type::YOLO26)
						image_based_boxes = cpu_nms(image_based_boxes, nms_threshold_);
					checkCudaRuntime(cudaStreamSynchronize(stream_));
					job.pro->set_value(image_based_boxes);
				}
				fetch_jobs.clear();
			}

			stream_ = nullptr;
			tensor_allocator_.reset();
			LOG_I("yoloseg", "Engine destroy.");
		}

		virtual bool preprocess(Job &job, const cv::Mat &image) override
		{
			if (tensor_allocator_ == nullptr)
			{
				LOG_E("yoloseg", "tensor_allocator_ is nullptr");
				return false;
			}

			if (image.empty())
			{
				LOG_E("yoloseg", "Image is empty");
				return false;
			}

			job.mono_tensor = tensor_allocator_->query();
			if (job.mono_tensor == nullptr)
			{
				LOG_E("yoloseg", "Tensor allocator query failed.");
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

			std::memcpy(image_host, image.data, size_image);
			std::memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
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
		{
			instance.reset();
		}
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

		std::memcpy(image_host, image.data, size_image);
		std::memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
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
