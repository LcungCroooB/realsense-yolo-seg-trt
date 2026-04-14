#include "d435_seg.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <tuple>
#include <utility>

#include "../common/utils.hpp"
#include "../logger/logger_macro.h"

namespace app_seg
{
	namespace
	{
		int clamp_to_int(float value, int low, int high)
		{
			if (low > high)
				std::swap(low, high);
			const int rounded = static_cast<int>(std::lround(value));
			return std::max(low, std::min(high, rounded));
		}

		std::string normalize_key(const std::string &text)
		{
			std::string normalized;
			normalized.reserve(text.size());
			for (char ch : text)
			{
				if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9'))
					normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
			}
			return normalized;
		}

		sensor::TimestampSource parse_timestamp_source(const std::string &text)
		{
			const std::string key = normalize_key(text);
			if (key == "globaldevice")
				return sensor::TimestampSource::kGlobalDevice;
			if (key == "deviceraw")
				return sensor::TimestampSource::kDeviceRaw;
			if (key == "hostreceive")
				return sensor::TimestampSource::kHostReceive;
			return sensor::TimestampSource::kUnknown;
		}

		bool parse_depth_strategy(const std::string &text, sensor::TargetDepthStrategy &strategy)
		{
			const std::string key = normalize_key(text);
			if (key == "bboxcenter")
			{
				strategy = sensor::TargetDepthStrategy::BBoxCenter;
				return true;
			}
			if (key == "maskmean")
			{
				strategy = sensor::TargetDepthStrategy::MaskMean;
				return true;
			}
			if (key == "maskmedian")
			{
				strategy = sensor::TargetDepthStrategy::MaskMedian;
				return true;
			}
			if (key == "maskedwindowmedian")
			{
				strategy = sensor::TargetDepthStrategy::MaskedWindowMedian;
				return true;
			}
			if (key == "erodedmaskmedian")
			{
				strategy = sensor::TargetDepthStrategy::ErodedMaskMedian;
				return true;
			}
			if (key == "maskpercentile")
			{
				strategy = sensor::TargetDepthStrategy::MaskPercentile;
				return true;
			}
			return false;
		}

		bool parse_pixel_strategy(const std::string &text, sensor::TargetPixelStrategy &strategy)
		{
			const std::string key = normalize_key(text);
			if (key == "bboxcenter")
			{
				strategy = sensor::TargetPixelStrategy::BBoxCenter;
				return true;
			}
			if (key == "maskcentroid")
			{
				strategy = sensor::TargetPixelStrategy::MaskCentroid;
				return true;
			}
			if (key == "maskmedian")
			{
				strategy = sensor::TargetPixelStrategy::MaskMedian;
				return true;
			}
			if (key == "maskinnerpoint")
			{
				strategy = sensor::TargetPixelStrategy::MaskInnerPoint;
				return true;
			}
			if (key == "erodedmaskcentroid")
			{
				strategy = sensor::TargetPixelStrategy::ErodedMaskCentroid;
				return true;
			}
			if (key == "nearestdepthtoz")
			{
				strategy = sensor::TargetPixelStrategy::NearestDepthToZ;
				return true;
			}
			return false;
		}

		sensor::D435Config build_sensor_config(const SegDepthConfig &config)
		{
			sensor::D435Config sensor_config;
			sensor_config.width = config.camera.width;
			sensor_config.height = config.camera.height;
			sensor_config.fps = config.camera.fps;
			sensor_config.serial = config.camera.serial;
			sensor_config.align_to_color = config.camera.align_to_color;
			sensor_config.enable_global_time = config.camera.enable_global_time;
			sensor_config.timestamp_source = parse_timestamp_source(config.camera.timestamp_source);
			sensor_config.auto_reconnect = config.camera.auto_reconnect;
			sensor_config.max_consecutive_failures = config.camera.max_consecutive_failures;
			return sensor_config;
		}

		bool build_target_config(const SegDepthConfig &config, sensor::TargetStrategyConfig &target_config, std::string &error)
		{
			error.clear();
			target_config.min_depth_m = config.depth.min_m;
			target_config.max_depth_m = config.depth.max_m;
			target_config.window_size = config.depth.window_size;
			target_config.erode_kernel = config.depth.erode_kernel;
			target_config.min_valid_samples = config.depth.min_valid_samples;
			target_config.trim_ratio = config.depth.trim_ratio;
			target_config.percentile = config.depth.percentile;

			if (!parse_depth_strategy(config.depth.z_strategy, target_config.depth_strategy))
			{
				error = utils::format(
					"Invalid depth.z_strategy='%s'. Valid: bbox_center, mask_mean, mask_median, masked_window_median, eroded_mask_median, mask_percentile",
					config.depth.z_strategy.c_str());
				return false;
			}

			if (!parse_pixel_strategy(config.depth.xy_strategy, target_config.pixel_strategy))
			{
				error = utils::format(
					"Invalid depth.xy_strategy='%s'. Valid: bbox_center, mask_centroid, mask_median, mask_inner_point, eroded_mask_centroid, nearest_depth_to_z",
					config.depth.xy_strategy.c_str());
				return false;
			}

			if (!parse_pixel_strategy(config.depth.xy_fallback_strategy, target_config.pixel_fallback_strategy))
			{
				error = utils::format(
					"Invalid depth.xy_fallback_strategy='%s'. Valid: bbox_center, mask_centroid, mask_median, mask_inner_point, eroded_mask_centroid, nearest_depth_to_z",
					config.depth.xy_fallback_strategy.c_str());
				return false;
			}

			return true;
		}

		bool class_allowed(const Modelconfig &config, int class_id)
		{
			if (!config.enable_class || config.class_ids.empty())
				return true;

			return std::find(config.class_ids.begin(), config.class_ids.end(), class_id) != config.class_ids.end();
		}

		cv::Rect clamp_rect_from_box(const yoloseg::Box &box, const cv::Size &image_size)
		{
			const int max_x = std::max(0, image_size.width - 1);
			const int max_y = std::max(0, image_size.height - 1);
			const int left = clamp_to_int(box.left, 0, max_x);
			const int top = clamp_to_int(box.top, 0, max_y);
			const int right = clamp_to_int(box.right, 0, max_x);
			const int bottom = clamp_to_int(box.bottom, 0, max_y);
			const int width = std::max(0, right - left);
			const int height = std::max(0, bottom - top);
			return cv::Rect(left, top, width, height);
		}

		std::pair<int, int> estimate_proto_size(const yoloseg::BoxArray &boxes)
		{
			int proto_width = 160;
			int proto_height = 160;
			for (const yoloseg::Box &box : boxes)
			{
				if (!box.seg)
					continue;
				proto_width = std::max(proto_width, box.seg->left + box.seg->width);
				proto_height = std::max(proto_height, box.seg->top + box.seg->height);
			}
			return std::make_pair(proto_width, proto_height);
		}

		cv::Mat decode_mask(const yoloseg::Box &box, const cv::Size &image_size, int proto_width, int proto_height)
		{
			if (!box.seg || image_size.width <= 0 || image_size.height <= 0 || proto_width <= 0 || proto_height <= 0)
				return cv::Mat();

			const int input_width = proto_width * 4;
			const int input_height = proto_height * 4;
			const float scale_x = static_cast<float>(input_width) / static_cast<float>(image_size.width);
			const float scale_y = static_cast<float>(input_height) / static_cast<float>(image_size.height);
			const float scale = std::min(scale_x, scale_y);
			const float ox = -scale * static_cast<float>(image_size.width) * 0.5f + static_cast<float>(input_width) * 0.5f + scale * 0.5f - 0.5f;
			const float oy = -scale * static_cast<float>(image_size.height) * 0.5f + static_cast<float>(input_height) * 0.5f + scale * 0.5f - 0.5f;

			const cv::Mat affine = (cv::Mat_<float>(2, 3) << scale, 0.0f, ox, 0.0f, scale, oy);
			cv::Mat inverse_affine;
			cv::invertAffineTransform(affine, inverse_affine);

			cv::Mat mask_map = cv::Mat::zeros(cv::Size(proto_width, proto_height), CV_8UC1);
			cv::Mat small_mask(box.seg->height, box.seg->width, CV_8UC1, box.seg->data);

			const int roi_x = std::max(0, std::min(box.seg->left, proto_width - 1));
			const int roi_y = std::max(0, std::min(box.seg->top, proto_height - 1));
			const int roi_w = std::min(box.seg->width, proto_width - roi_x);
			const int roi_h = std::min(box.seg->height, proto_height - roi_y);
			if (roi_w <= 0 || roi_h <= 0)
				return cv::Mat();

			small_mask(cv::Rect(0, 0, roi_w, roi_h)).copyTo(mask_map(cv::Rect(roi_x, roi_y, roi_w, roi_h)));
			cv::resize(mask_map, mask_map, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);
			cv::threshold(mask_map, mask_map, 128, 255, cv::THRESH_BINARY);

			cv::Mat mask_resized;
			cv::warpAffine(mask_map, mask_resized, inverse_affine, image_size, cv::INTER_LINEAR);
			cv::threshold(mask_resized, mask_resized, 1, 255, cv::THRESH_BINARY);
			return mask_resized;
		}

		void draw_mask_overlay(cv::Mat &canvas, const cv::Mat &mask, const cv::Scalar &color)
		{
			if (canvas.empty() || mask.empty())
				return;

			cv::Mat colored(canvas.size(), CV_8UC3, color);
			cv::Mat blended;
			cv::addWeighted(canvas, 0.65, colored, 0.35, 0.0, blended);
			blended.copyTo(canvas, mask);
		}
	}

	D435SegApp::D435SegApp() = default;

	D435SegApp::~D435SegApp()
	{
		shutdown();
	}

	bool D435SegApp::init(const SegDepthConfig &config)
	{
		shutdown();

		config_ = config;
		frame_index_ = 0;
		initialized_ = false;
		running_ = false;

		if (!sensor_.open(build_sensor_config(config_)))
		{
			LOG_E("seg", "Failed to open D435 sensor");
			return false;
		}

		if (!sensor_.get_camera_parameters(camera_parameters_))
		{
			LOG_E("seg", "Failed to query D435 camera parameters");
			sensor_.close();
			return false;
		}

		projector_.set_camera_parameters(camera_parameters_);
		sensor::TargetStrategyConfig target_config;
		std::string strategy_error;
		if (!build_target_config(config_, target_config, strategy_error))
		{
			LOG_E("seg", "%s", strategy_error.c_str());
			sensor_.close();
			return false;
		}
		target_selector_.reset(new sensor::TargetSelector(target_config));

		if (config_.model.enabled)
		{
			infer_ = yoloseg::create_infer(
				config_.model.engine_path,
				yoloseg::type_from_string(config_.model.yolo_type),
				config_.model.gpu_id,
				config_.model.confidence_threshold,
				config_.model.nms_threshold,
				yoloseg::NMSMethod::FastGPU,
				config_.model.topk,
				false);
			if (!infer_)
			{
				LOG_E("seg", "Failed to create seg infer from %s", config_.model.engine_path.c_str());
				sensor_.close();
				return false;
			}

			const int warmup = std::max(0, config_.runtime.warmup);
			if (warmup > 0)
			{
				cv::Mat warmup_image(config_.camera.height, config_.camera.width, CV_8UC3, cv::Scalar(114, 114, 114));
				for (int i = 0; i < warmup; ++i)
					infer_->commit(warmup_image).get();
			}
		}

		initialized_ = true;
		LOG_I("seg", "D435SegApp initialized (model=%s, warmup=%d)",
			  config_.model.enabled ? config_.model.engine_path.c_str() : "disabled",
			  config_.runtime.warmup);
		return true;
	}

	void D435SegApp::shutdown()
	{
		running_ = false;
		infer_.reset();
		target_selector_.reset();
		sensor_.close();
		initialized_ = false;
		frame_index_ = 0;
	}

	bool D435SegApp::is_ready() const
	{
		return initialized_;
	}

	bool D435SegApp::is_running() const
	{
		return running_;
	}

	const SegDepthConfig *D435SegApp::config() const
	{
		return initialized_ ? &config_ : nullptr;
	}

	bool D435SegApp::get_camera_parameters(sensor::D435CameraParameters &camera_parameters) const
	{
		if (!initialized_)
			return false;
		camera_parameters = camera_parameters_;
		return true;
	}

	bool D435SegApp::read_and_process(SegFrameResult &result)
	{
		if (!initialized_)
			return false;

		sensor::D435Frame frame;
		if (!sensor_.read(frame, config_.runtime.read_timeout_ms))
			return false;

		return process_frame(frame, frame_index_++, result);
	}

	bool D435SegApp::process_frame(const sensor::D435Frame &frame, int frame_index, SegFrameResult &result)
	{
		result = SegFrameResult();
		if (!initialized_ || !frame.valid())
			return false;

		result.frame_index = frame_index;
		result.timestamp_ms = frame.timestamp_ms;
		result.serial = frame.serial;
		result.color = frame.color.clone();
		result.depth = frame.depth.clone();
		result.depth_meters = frame.depth.clone();
		result.valid = true;

		if (!config_.model.enabled || !infer_)
			return true;

		yoloseg::BoxArray boxes;
		try
		{
			boxes = infer_->commit(frame.color).get();
		}
		catch (const std::exception &e)
		{
			LOG_E("seg", "Infer failed on frame %d: %s", frame_index, e.what());
			return false;
		}
		const std::pair<int, int> proto_size = estimate_proto_size(boxes);

		std::size_t object_index = 0;
		for (const yoloseg::Box &box : boxes)
		{
			if (!class_allowed(config_.model, box.class_label))
				continue;

			SegObjectResult object;
			object.index = object_index++;
			object.class_id = box.class_label;
			object.confidence = box.confidence;
			object.bbox = clamp_rect_from_box(box, frame.color.size());
			object.mask = decode_mask(box, frame.color.size(), proto_size.first, proto_size.second);

			if (object.bbox.width <= 0 || object.bbox.height <= 0)
			{
				result.objects.emplace_back(std::move(object));
				continue;
			}

			if (target_selector_)
			{
				const bool selected = target_selector_->select(result.depth_meters, object.mask, object.bbox);
				object.target = target_selector_->result();
				if (selected)
				{
					cv::Point3f point3d;
					if (projector_.deproject_color_pixel(object.target.pixel.x, object.target.pixel.y, object.target.depth_m, point3d))
					{
						object.target.point3d = cv::Point3d(
							static_cast<double>(point3d.x),
							static_cast<double>(point3d.y),
							static_cast<double>(point3d.z));
					}
				}
			}

			result.objects.emplace_back(std::move(object));
		}

		return true;
	}

	bool D435SegApp::render(const SegFrameResult &result, cv::Mat &canvas) const
	{
		if (result.color.empty())
			return false;

		canvas = result.color.clone();
		for (const SegObjectResult &object : result.objects)
		{
			const std::tuple<uint8_t, uint8_t, uint8_t> color = utils::random_color(object.class_id);
			const cv::Scalar bgr(
				static_cast<int>(std::get<0>(color)),
				static_cast<int>(std::get<1>(color)),
				static_cast<int>(std::get<2>(color)));

			draw_mask_overlay(canvas, object.mask, bgr);
			if (object.bbox.width > 0 && object.bbox.height > 0)
				cv::rectangle(canvas, object.bbox, bgr, 2, cv::LINE_AA);

			const std::string label = utils::format("id=%d conf=%.2f", object.class_id, object.confidence);
			const cv::Point label_origin(object.bbox.x, std::max(18, object.bbox.y - 6));
			cv::putText(canvas, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2, cv::LINE_AA);

			if (object.target.valid)
			{
				cv::circle(canvas, object.target.pixel, 4, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
				const std::string depth_text = utils::format(
					"z=%.3fm xyz=(%.3f, %.3f, %.3f)",
					object.target.depth_m,
					object.target.point3d.x,
					object.target.point3d.y,
					object.target.point3d.z);
				cv::putText(canvas,
							depth_text,
							cv::Point(object.bbox.x, std::min(canvas.rows - 8, object.bbox.y + object.bbox.height + 18)),
							cv::FONT_HERSHEY_SIMPLEX,
							0.5,
							cv::Scalar(0, 255, 255),
							1,
							cv::LINE_AA);
			}
		}

		cv::putText(canvas,
					utils::format("frame=%d objects=%d ts=%llu", result.frame_index, static_cast<int>(result.objects.size()), static_cast<unsigned long long>(result.timestamp_ms)),
					cv::Point(12, 24),
					cv::FONT_HERSHEY_SIMPLEX,
					0.65,
					cv::Scalar(0, 255, 0),
					2,
					cv::LINE_AA);

		if (config_.show.show_camera_axes_info)
		{
			const std::string camera_text = utils::format(
				"serial=%s fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
				result.serial.c_str(),
				camera_parameters_.color_intrinsics.fx,
				camera_parameters_.color_intrinsics.fy,
				camera_parameters_.color_intrinsics.cx,
				camera_parameters_.color_intrinsics.cy);
			cv::putText(canvas,
						camera_text,
						cv::Point(12, 48),
						cv::FONT_HERSHEY_SIMPLEX,
						0.55,
						cv::Scalar(255, 255, 0),
						1,
						cv::LINE_AA);
		}

		return true;
	}

	int D435SegApp::run()
	{
		if (!initialized_)
			return -1;

		running_ = true;
		const int max_frames = config_.runtime.max_frames;
		const int max_consecutive_failures = std::max(1, config_.camera.max_consecutive_failures);
		int consecutive_failures = 0;

		while (running_)
		{
			if (max_frames > 0 && frame_index_ >= max_frames)
				break;

			SegFrameResult result;
			if (!read_and_process(result))
			{
				++consecutive_failures;
				LOG_W("seg", "Failed to read/process frame (consecutive_failures=%d/%d)",
					  consecutive_failures,
					  max_consecutive_failures);
				if (consecutive_failures >= max_consecutive_failures)
				{
					LOG_E("seg", "Too many consecutive failures, stop app");
					running_ = false;
					return -1;
				}
				continue;
			}

			consecutive_failures = 0;

			if (config_.show.enable)
			{
				cv::Mat canvas;
				if (render(result, canvas))
				{
					cv::imshow(config_.show.window_name, canvas);
					const int key = cv::waitKey(std::max(1, config_.show.wait_key_ms));
					if (key == 27 || key == 'q' || key == 'Q')
						break;
				}
			}
		}

		running_ = false;
		if (config_.show.enable)
			cv::destroyWindow(config_.show.window_name);
		return 0;
	}

	int app_d435_seg(const std::string &config_path)
	{
		SegDepthConfig config;
		if (!load_seg_depth_config(config_path, config))
			return -1;

		D435SegApp app;
		if (!app.init(config))
			return -1;

		return app.run();
	}
}
