#pragma once

#include <string>
#include <vector>

namespace app_seg
{
    struct Cameraconfig
    {
        int width = 640;
        int height = 480;
        int fps = 30;
        bool align_to_color = true;
        bool enable_global_time = false;
        std::string timestamp_source = "global_device";
        bool auto_reconnect = true;
        int max_consecutive_failures = 5;
        std::string serial;
    };

    struct Videoconfig
    {
        bool loop = true;
    };

    struct Modelconfig
    {
        bool enabled = true;
        std::string engine_path;
        std::string yolo_type = "v11";
        int gpu_id = 0;
        float confidence_threshold = 0.25f;
        float nms_threshold = 0.45f;
        int topk = 10;

        bool enable_class = false;
        std::vector<int> class_ids;
    };

    struct Runtimeconfig
    {
        int warmup = 5;
        int batch_size = 1;
        int max_frames = -1;
        int read_timeout_ms = 1000;
    };

    struct Depthconfig
    {
        float min_m = 0.10f;
        float max_m = 4.0f;
        std::string z_strategy;
        std::string xy_strategy;
        std::string xy_fallback_strategy;

        int window_size = 5;
        int erode_kernel = 5;
        int min_valid_samples = 5;
        float trim_ratio = 0.0f;
        float percentile = 0.3f;
    };

    struct Showconfig
    {
        bool enable = true;
        std::string window_name = "seg_depth";
        bool show_camera_axes_info = false;
        int wait_key_ms = 1;
    };

    struct SegDepthConfig
    {
        Cameraconfig camera;
        Videoconfig video;
        Modelconfig model;
        Runtimeconfig runtime;
        Depthconfig depth;
        Showconfig show;
    };

    bool load_seg_depth_config(const std::string &config_path, SegDepthConfig &config);
}