#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

namespace sensor
{
    enum class D435State // 相机状态
    {
        kClosed = 0,
        kOpened = 1,
        kStreaming = 2,
        kError = 3,
    };

    enum class TimestampSource // 时间来源
    {
        kUnknown = 0,
        kGlobalDevice = 1, // 设备全局时间戳，适用于多设备同步
        kDeviceRaw = 2,    // 设备原始时间戳，适用于单设备使用
        kHostReceive = 3,  // 主机接收时间戳，适用于对时间精度要求不高的场景
    };

    struct D435Intrinsics // 相机内参
    {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        int width = 0;
        int height = 0;
        std::array<float, 5> coeffs = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    };

    struct D435Extrinsics // 相机外参
    {
        std::array<float, 9> rotation = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        std::array<float, 3> translation = {0.0f, 0.0f, 0.0f};
    };

    struct D435CameraParameters
    {
        float depth_scale = 0.001f; // 深度单位转换为米的比例
        D435Intrinsics color_intrinsics;
        D435Intrinsics depth_intrinsics;
        D435Extrinsics depth_to_color;
    };

    struct D435Frame // 相机帧数据
    {
        cv::Mat color;
        cv::Mat depth;
        bool depth_aligned_to_color = true;
        std::uint64_t timestamp_ms = 0;
        std::string serial;
        bool valid() const { return !color.empty() && !depth.empty(); }
    };

    struct D435Config
    {
        int width = 640;
        int height = 480;
        int fps = 30;
        std::string serial;
        // Standard output strategy for read(D435Frame&, D435ReadOptions).
        bool align_to_color = true;
        // Enable RealSense global-time mapping when supported by the device.
        bool enable_global_time = false;
        TimestampSource timestamp_source = TimestampSource::kUnknown;
        // Automatically attempt to reconnect if the camera connection is lost during streaming.
        bool auto_reconnect = true;
        int max_consecutive_failures = 5;
    };

    class D435Sensor
    {
    public:
        D435Sensor();
        ~D435Sensor();

        bool open(const D435Config &config);
        void close();
        bool is_opened() const;
        // Returns true if a new frame was successfully read within the specified timeout, false otherwise.
        bool read(D435Frame &frame, int timeout_ms = 1000);
        bool get_camera_parameters(D435CameraParameters &camera_parameters) const;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}