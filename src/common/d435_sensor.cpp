#include "d435_sensor.hpp"

#include <cmath>
#include "utils.hpp"
#include "../logger/logger_macro.h"

namespace sensor
{
    namespace
    {
        D435Intrinsics extract_intrinsics(const rs2_intrinsics &in)
        {
            D435Intrinsics intrinsics;
            intrinsics.fx = in.fx;
            intrinsics.fy = in.fy;
            intrinsics.cx = in.ppx;
            intrinsics.cy = in.ppy;
            intrinsics.width = in.width;
            intrinsics.height = in.height;
            for (int i = 0; i < 5; ++i)
                intrinsics.coeffs[i] = in.coeffs[i];
            return intrinsics;
        }

        D435Extrinsics extract_extrinsics(const rs2_extrinsics &ex)
        {
            D435Extrinsics extrinsics;
            for (int i = 0; i < 9; ++i)
                extrinsics.rotation[i] = ex.rotation[i];
            for (int i = 0; i < 3; ++i)
                extrinsics.translation[i] = ex.translation[i];
            return extrinsics;
        }

        std::uint64_t pick_timestamp_ms(TimestampSource source, std::uint64_t host_ts, double device_ts, rs2_timestamp_domain domain)
        {
            const bool valid_device_ts = std::isfinite(device_ts) && device_ts > 0.0;
            if (source == TimestampSource::kDeviceRaw && valid_device_ts)
            {
                return static_cast<std::uint64_t>(device_ts);
            }
            if (source == TimestampSource::kGlobalDevice && valid_device_ts && domain == RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME)
            {
                return static_cast<std::uint64_t>(device_ts);
            }
            return host_ts;
        }
    }

    struct D435Sensor::Impl
    {
        mutable std::mutex mutex;
        bool opened = false;
        int failures = 0;
        D435State state = D435State::kClosed;
        D435Config config;
        D435CameraParameters camera_parameters;
        std::string serial;
        rs2::pipeline pipeline;
        rs2::config rs_config;
        rs2::align align_to_color = rs2::align(RS2_STREAM_COLOR);
        rs2::pipeline_profile profile;

    public:
        bool start_pipeline()
        {
            profile = pipeline.start(rs_config);
            rs2::device device = profile.get_device();
            rs2::depth_sensor depth_sensor = device.first<rs2::depth_sensor>();

            serial = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
            camera_parameters.depth_scale = depth_sensor.get_depth_scale();
            if (depth_sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED))
                depth_sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, config.enable_global_time ? 1.0f : 0.0f);

            rs2::video_stream_profile color = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
            rs2::video_stream_profile depth = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            camera_parameters.color_intrinsics = extract_intrinsics(color.get_intrinsics());
            camera_parameters.depth_intrinsics = extract_intrinsics(depth.get_intrinsics());
            camera_parameters.depth_to_color = extract_extrinsics(depth.get_extrinsics_to(color));

            opened = true;
            failures = 0;
            state = D435State::kStreaming;
            return true;
        };
    };

    D435Sensor::D435Sensor() : impl_(std::make_unique<Impl>()) {}
    D435Sensor::~D435Sensor() { close(); }

    bool D435Sensor::open(const D435Config &config)
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->state = D435State::kOpened;
        if (impl_->opened)
            impl_->pipeline.stop();
        try
        {
            impl_->config = config;
            impl_->rs_config = rs2::config();
            impl_->rs_config.enable_stream(RS2_STREAM_COLOR, config.width, config.height, RS2_FORMAT_BGR8, config.fps);
            impl_->rs_config.enable_stream(RS2_STREAM_DEPTH, config.width, config.height, RS2_FORMAT_Z16, config.fps);
            if (!config.serial.empty())
                impl_->rs_config.enable_device(config.serial);

            impl_->start_pipeline();
            LOG_I("sensor", "D435 opened: serial=%s, %dx%d@%d", impl_->serial.c_str(), config.width, config.height, config.fps);
            return true;
        }
        catch (const std::exception &e)
        {
            impl_->opened = false;
            impl_->state = D435State::kError;
            LOG_E("sensor", "Failed to open D435: %s", e.what());
            return false;
        }
    };

    void D435Sensor::close()
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->opened)
            return;
        try
        {
            impl_->pipeline.stop();
        }
        catch (...)
        {
            LOG_W("sensor", "D435 stop raised exception");
        }
        impl_->opened = false;
        impl_->state = D435State::kClosed;
    };

    bool D435Sensor::is_opened() const
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->opened;
    }

    bool D435Sensor::get_camera_parameters(D435CameraParameters &camera_parameters) const
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->opened)
            return false;
        camera_parameters = impl_->camera_parameters;
        return true;
    }

    bool D435Sensor::read(D435Frame &frame, int timeout_ms)
    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        if (!impl_->opened)
            return false;

        const D435Config cfg = impl_->config;
        const float depth_scale = impl_->camera_parameters.depth_scale;
        const bool align = cfg.align_to_color;
        if (!std::isfinite(depth_scale) || depth_scale <= 0.0f)
            return false;

        lock.unlock();

        try
        {
            rs2::frameset frameset = impl_->pipeline.wait_for_frames(timeout_ms);
            if (!frameset)
                throw std::runtime_error("Timeout waiting for frames");
            if (align)
                frameset = impl_->align_to_color.process(frameset);

            rs2::video_frame color_frame = frameset.get_color_frame();
            rs2::depth_frame depth_frame = frameset.get_depth_frame();
            if (!color_frame || !depth_frame)
                throw std::runtime_error("Failed to get color or depth frame");

            cv::Mat color_wrap(color_frame.get_height(), color_frame.get_width(), CV_8UC3, const_cast<void *>(color_frame.get_data()), cv::Mat::AUTO_STEP);
            cv::Mat depth_raw(depth_frame.get_height(), depth_frame.get_width(), CV_16UC1, const_cast<void *>(depth_frame.get_data()), cv::Mat::AUTO_STEP);
            cv::Mat depth_f32;
            depth_raw.convertTo(depth_f32, CV_32FC1, depth_scale);

            const std::uint64_t host_ts = static_cast<std::uint64_t>(utils::time::timestamp_millisecond());
            const double dev_ts = depth_frame.get_timestamp();
            const rs2_timestamp_domain ts_domain = depth_frame.get_frame_timestamp_domain();

            frame.color = color_wrap.clone();
            frame.depth = std::move(depth_f32);
            frame.depth_aligned_to_color = align;
            frame.timestamp_ms = pick_timestamp_ms(cfg.timestamp_source, host_ts, dev_ts, ts_domain);

            {
                std::lock_guard<std::mutex> state_lock(impl_->mutex);
                if (!impl_->opened)
                    return false;
                frame.serial = impl_->serial;
                impl_->failures = 0; // reset failure count on successful read
                impl_->state = D435State::kStreaming;
            }
            return true;
        }
        catch (const std::exception &e)
        {
            bool need_reopen = false;
            {
                std::lock_guard<std::mutex> fail_lock(impl_->mutex);
                impl_->state = D435State::kError;
                impl_->failures += 1;
                need_reopen = impl_->opened && cfg.auto_reconnect && cfg.max_consecutive_failures > 0 && impl_->failures >= cfg.max_consecutive_failures;
            }
            LOG_W("sensor", "Failed to read frame: %s", e.what());
            if (!need_reopen)
                return false;

            std::lock_guard<std::mutex> reopen_lock(impl_->mutex);
            if (!impl_->opened)
                return false;
            try
            {
                return impl_->start_pipeline();
            }
            catch (const std::exception &open_error)
            {
                impl_->opened = false;
                impl_->state = D435State::kError;
                LOG_E("sensor", "Failed to reopen D435: %s", open_error.what());
                return false;
            }
        }
    };

}