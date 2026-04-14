#include "d435_seg_config.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <yaml-cpp/yaml.h>

#include "yaml_helpers.hpp"
#include "../common/utils.hpp"
#include "../logger/logger_macro.h"


namespace app_seg
{
	bool load_seg_depth_config(const std::string &config_path, SegDepthConfig &config)
	{
		std::string resolved_config;
		std::string config_prefix;
		if (!utils::path::resolve_input_path(config_path, resolved_config, config_prefix))
		{
			LOG_E("seg", "Seg-depth config not found: %s", config_path.c_str());
			return false;
		}

		YAML::Node root;
		try
		{
			root = YAML::LoadFile(resolved_config);
		}
		catch (const YAML::Exception &e)
		{
			LOG_E("seg", "Failed to parse seg-depth config '%s': %s", resolved_config.c_str(), e.what());
			return false;
		}

		const YAML::Node camera_node = root["camera"];
		config.camera.width = yaml_node::read_int(camera_node, "width", config.camera.width);
		config.camera.height = yaml_node::read_int(camera_node, "height", config.camera.height);
		config.camera.fps = yaml_node::read_int(camera_node, "fps", config.camera.fps);
		config.camera.align_to_color = yaml_node::read_bool(camera_node, "align_to_color", config.camera.align_to_color);
		config.camera.enable_global_time = yaml_node::read_bool(camera_node, "enable_global_time", config.camera.enable_global_time);
		config.camera.timestamp_source = yaml_node::read_str(camera_node, "timestamp_source", config.camera.timestamp_source);
		config.camera.auto_reconnect = yaml_node::read_bool(camera_node, "auto_reconnect", config.camera.auto_reconnect);
		config.camera.max_consecutive_failures = yaml_node::read_int(camera_node, "max_consecutive_failures", config.camera.max_consecutive_failures);
		config.camera.serial = yaml_node::read_str(
			camera_node,
			"serial",
			yaml_node::read_str(camera_node, "serial_number", config.camera.serial));

		const YAML::Node video_node = root["video"];
		config.video.loop = yaml_node::read_bool(video_node, "loop", config.video.loop);

		const YAML::Node model_node = root["model"];
		config.model.enabled = yaml_node::read_bool(model_node, "enabled", config.model.enabled);
		config.model.yolo_type = yaml_node::read_str(model_node, "yolo_type", config.model.yolo_type);
		config.model.gpu_id = yaml_node::read_int(model_node, "gpu_id", config.model.gpu_id);
		config.model.confidence_threshold = yaml_node::read_float(model_node, "confidence_threshold", config.model.confidence_threshold);
		config.model.nms_threshold = yaml_node::read_float(model_node, "nms_threshold", config.model.nms_threshold);
		config.model.topk = yaml_node::read_int(model_node, "topk", config.model.topk);

		const std::string engine_path_raw = yaml_node::read_str(
			model_node,
			"engine_path",
			yaml_node::read_str(model_node, "engine_file", config.model.engine_path));

		if (config.model.enabled)
		{
			if (engine_path_raw.empty())
			{
				LOG_E("seg", "Config '%s' must define model.engine_file or model.engine_path when model is enabled", resolved_config.c_str());
				return false;
			}

			std::string resolved_engine;
			std::string engine_prefix;
			if (!utils::path::resolve_input_path(engine_path_raw, resolved_engine, engine_prefix))
			{
				LOG_E("seg", "Seg model engine not found: %s", engine_path_raw.c_str());
				return false;
			}
			config.model.engine_path = resolved_engine;
		}
		else
		{
			config.model.engine_path = utils::path::resolve_output_path(engine_path_raw, config_prefix);
		}

		const YAML::Node class_filter_node = model_node["class_filter"];
		config.model.enable_class = yaml_node::read_bool(
			class_filter_node,
			"enable",
			yaml_node::read_bool(model_node, "enable_class", config.model.enable_class));

		if (class_filter_node && class_filter_node["class_ids"])
			config.model.class_ids = yaml_node::read_int_list(class_filter_node, "class_ids");
		else if (model_node["class_ids"])
			config.model.class_ids = yaml_node::read_int_list(model_node, "class_ids");

		const YAML::Node runtime_node = root["runtime"];
		config.runtime.warmup = yaml_node::read_int(runtime_node, "warmup", config.runtime.warmup);
		config.runtime.batch_size = yaml_node::read_int(runtime_node, "batch_size", config.runtime.batch_size);
		config.runtime.max_frames = yaml_node::read_int(runtime_node, "max_frames", config.runtime.max_frames);
		config.runtime.read_timeout_ms = yaml_node::read_int(runtime_node, "read_timeout_ms", config.runtime.read_timeout_ms);

		const YAML::Node depth_node = root["depth"];
		config.depth.min_m = yaml_node::read_float(depth_node, "min_m", config.depth.min_m);
		config.depth.max_m = yaml_node::read_float(depth_node, "max_m", config.depth.max_m);
		config.depth.z_strategy = yaml_node::read_str(depth_node, "z_strategy", config.depth.z_strategy);
		config.depth.xy_strategy = yaml_node::read_str(depth_node, "xy_strategy", config.depth.xy_strategy);
		config.depth.xy_fallback_strategy = yaml_node::read_str(depth_node, "xy_fallback_strategy", config.depth.xy_fallback_strategy);
		config.depth.window_size = yaml_node::read_int(depth_node, "window_size", config.depth.window_size);
		config.depth.erode_kernel = yaml_node::read_int(depth_node, "erode_kernel", config.depth.erode_kernel);
		config.depth.min_valid_samples = yaml_node::read_int(depth_node, "min_valid_samples", config.depth.min_valid_samples);
		config.depth.trim_ratio = yaml_node::read_float(depth_node, "trim_ratio", config.depth.trim_ratio);
		config.depth.percentile = yaml_node::read_float(depth_node, "percentile", config.depth.percentile);

		const YAML::Node show_node = root["show"];
		config.show.enable = yaml_node::read_bool(show_node, "enable", config.show.enable);
		config.show.window_name = yaml_node::read_str(show_node, "window_name", config.show.window_name);
		config.show.show_camera_axes_info = yaml_node::read_bool(show_node, "show_camera_axes_info", config.show.show_camera_axes_info);
		config.show.wait_key_ms = yaml_node::read_int(show_node, "wait_key_ms", config.show.wait_key_ms);

		if (config.runtime.batch_size <= 0)
			config.runtime.batch_size = 1;
		if (config.runtime.read_timeout_ms < 0)
			config.runtime.read_timeout_ms = 0;
		if (config.camera.max_consecutive_failures <= 0)
			config.camera.max_consecutive_failures = 1;
		if (config.depth.min_valid_samples < 0)
			config.depth.min_valid_samples = 0;
		if (config.depth.window_size <= 0)
			config.depth.window_size = 1;
		if (config.depth.erode_kernel < 0)
			config.depth.erode_kernel = 0;
		if (config.depth.min_m < 0.0f)
			config.depth.min_m = 0.0f;
		if (config.depth.max_m > 0.0f && config.depth.max_m < config.depth.min_m)
			std::swap(config.depth.min_m, config.depth.max_m);
		if (config.depth.trim_ratio < 0.0f)
			config.depth.trim_ratio = 0.0f;
		if (config.depth.trim_ratio > 0.49f)
			config.depth.trim_ratio = 0.49f;
		if (config.depth.percentile < 0.0f)
			config.depth.percentile = 0.0f;
		if (config.depth.percentile > 1.0f)
			config.depth.percentile = 1.0f;
		if (config.model.topk <= 0)
			config.model.topk = 1;

		LOG_I("seg",
			  "Loaded seg-depth config from %s (engine=%s, camera=%dx%d@%d, batch=%d)",
			  resolved_config.c_str(),
			  config.model.engine_path.c_str(),
			  config.camera.width,
			  config.camera.height,
			  config.camera.fps,
			  config.runtime.batch_size);
		return true;
	}
}
