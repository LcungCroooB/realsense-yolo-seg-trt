#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "d435_seg_config.hpp"
#include "../common/d435_geometry.hpp"
#include "../common/d435_sensor.hpp"
#include "../common/target_strategy.hpp"
#include "../tasks/yolo_seg.hpp"

namespace app_seg
{
	struct SegObjectResult
	{
		std::size_t index = 0;
		int class_id = -1;
		float confidence = 0.0f;
		cv::Rect bbox;
		cv::Mat mask;
		sensor::TargetStrategyResult target;
	};

	struct SegFrameResult
	{
		bool valid = false;
		int frame_index = -1;
		std::uint64_t timestamp_ms = 0;
		std::string serial;
		cv::Mat color;
		cv::Mat depth;
		cv::Mat depth_meters;
		std::vector<SegObjectResult> objects;

		bool empty() const
		{
			return color.empty() || depth.empty();
		}
	};

	class D435SegApp
	{
	public:
		D435SegApp();
		~D435SegApp();

		bool init(const SegDepthConfig &config);
		void shutdown();

		bool is_ready() const;
		bool is_running() const;

		const SegDepthConfig *config() const;
		bool get_camera_parameters(sensor::D435CameraParameters &camera_parameters) const;

		bool read_and_process(SegFrameResult &result);
		bool process_frame(const sensor::D435Frame &frame, int frame_index, SegFrameResult &result);
		bool render(const SegFrameResult &result, cv::Mat &canvas) const;

		int run();

	private:
		SegDepthConfig config_;
		sensor::D435Sensor sensor_;
		sensor::D435CameraParameters camera_parameters_;
		sensor::D435GeometryProjector projector_;
		std::shared_ptr<yoloseg::Infer> infer_;
		std::unique_ptr<sensor::TargetSelector> target_selector_;
		bool initialized_ = false;
		bool running_ = false;
		int frame_index_ = 0;
	};

	int app_d435_seg(const std::string &config_path);
}
