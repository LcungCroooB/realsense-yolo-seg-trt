#include "target_strategy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../logger/logger_macro.h"

namespace sensor
{
    namespace
    {
        // Check if the depth map is of type CV_32FC1 (32-bit float, single channel)
        bool is_depth_map_32f(const cv::Mat &depth_meters)
        {
            return !depth_meters.empty() && depth_meters.type() == CV_32FC1;
        }

        // Check if the mask is of type CV_8UC1 (8-bit unsigned char, single channel)
        bool is_mask_u8(const cv::Mat &mask)
        {
            return !mask.empty() && mask.type() == CV_8UC1;
        }

        // Check if the depth map and mask have the same size
        bool mask_matches_depth(const cv::Mat &depth_meters, const cv::Mat &mask)
        {
            return !depth_meters.empty() && !mask.empty() && depth_meters.size() == mask.size();
        }

        // Check if the bounding box is valid (positive width and height)
        bool valid_bbox(const cv::Rect &bbox)
        {
            return bbox.width > 0 && bbox.height > 0;
        }

        // Clamp the bounding box to the image boundaries and check if the resulting rectangle is valid
        bool clamp_rect_to_image(const cv::Rect &input, const cv::Size &image_size, cv::Rect &clamped)
        {
            const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
            clamped = input & image_rect;
            return valid_bbox(clamped);
        }

        // Normalize an odd kernel size by ensuring it is positive and odd. If the input is even, increment it by 1 to make it odd.
        bool normalize_odd_kernel(int input, int &output)
        {
            if (input <= 0)
                return false;
            output = (input % 2 == 0) ? (input + 1) : input;
            return true;
        }

        // Check if a depth value is valid (finite, positive, and within the specified range)
        bool depth_in_range(float depth_m, const float &min_depth_m, const float &max_depth_m)
        {
            if (!std::isfinite(depth_m) || depth_m <= 0.0f)
                return false;
            return depth_m >= min_depth_m && depth_m <= max_depth_m;
        }

        // get mask values within a rectangle, optionally applying a mask to filter out invalid pixels
        template <typename F>
        bool for_each_mask_pixel_in_roi(const cv::Mat &mask, const cv::Rect &roi, F &&func)
        {
            cv::Rect bounded;
            if (!clamp_rect_to_image(roi, mask.size(), bounded))
                return false;

            bool any = false;
            for (int y = bounded.y; y < bounded.y + bounded.height; ++y)
            {
                const unsigned char *row = mask.ptr<unsigned char>(y);
                for (int x = bounded.x; x < bounded.x + bounded.width; ++x)
                {
                    if (row[x] == 0)
                        continue;
                    func(x, y);
                    any = true;
                }
            }
            return any;
        }

        // get the center point of a rectangle
        cv::Point roi_center(const cv::Rect &roi)
        {
            return cv::Point(roi.x + roi.width / 2, roi.y + roi.height / 2);
        }

        bool pick_bbox_center(const cv::Rect &roi, cv::Point &pixel)
        {
            if (!valid_bbox(roi))
                return false;
            pixel = roi_center(roi);
            return true;
        }

        bool pick_mask_centroid(const cv::Mat &mask, const cv::Rect &roi, cv::Point &pixel)
        {
            if (mask.empty())
                return false;
            double sum_x = 0.0, sum_y = 0.0;
            int count = 0;
            auto func = [&](int x, int y)
            {
                sum_x += static_cast<double>(x);
                sum_y += static_cast<double>(y);
                ++count;
            };
            if (!for_each_mask_pixel_in_roi(mask, roi, func) || count <= 0)
                return false;

            pixel.x = static_cast<int>(std::lround(sum_x / static_cast<double>(count)));
            pixel.y = static_cast<int>(std::lround(sum_y / static_cast<double>(count)));
            return true;
        }

        bool pick_mask_median(const cv::Mat &mask, const cv::Rect &roi, cv::Point &pixel)
        {
            if (mask.empty())
                return false;
            std::vector<int> xs, ys;
            auto func = [&](int x, int y)
            {
                xs.push_back(x);
                ys.push_back(y);
            };
            if (!for_each_mask_pixel_in_roi(mask, roi, func) || xs.empty())
                return false;

            const std::size_t mid = xs.size() / 2;
            std::nth_element(xs.begin(), xs.begin() + static_cast<long>(mid), xs.end());
            std::nth_element(ys.begin(), ys.begin() + static_cast<long>(mid), ys.end());
            pixel.x = xs[mid];
            pixel.y = ys[mid];
            return true;
        }

        bool pick_mask_inner_point(const cv::Mat &mask, const cv::Rect &roi, cv::Point &pixel)
        {
            if (mask.empty())
                return false;

            cv::Rect bounded;
            if (!clamp_rect_to_image(roi, mask.size(), bounded))
                return false;

            const cv::Mat roi_mask = mask(bounded);
            if (cv::countNonZero(roi_mask) <= 0)
                return false;

            cv::Mat distance;
            cv::distanceTransform(roi_mask, distance, cv::DIST_L2, 3);
            double max_val = 0.0;
            cv::Point max_loc;
            cv::minMaxLoc(distance, nullptr, &max_val, nullptr, &max_loc);
            if (max_val <= 0.0)
                return false;

            pixel = cv::Point(max_loc.x + bounded.x, max_loc.y + bounded.y);
            return true;
        }

        bool pick_eroded_mask_centroid(const cv::Mat &mask, const cv::Rect &roi, int erode_kernel, cv::Point &pixel)
        {
            if (mask.empty())
                return false;

            cv::Rect bounded;
            if (!clamp_rect_to_image(roi, mask.size(), bounded))
                return false;

            int kernel = 0;
            if (!normalize_odd_kernel(erode_kernel, kernel))
                return false;

            cv::Mat eroded;
            cv::erode(mask(bounded), eroded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel, kernel)));
            if (cv::countNonZero(eroded) <= 0)
                return false;

            cv::Point local_centroid;
            if (!pick_mask_centroid(eroded, cv::Rect(0, 0, eroded.cols, eroded.rows), local_centroid))
                return false;

            // local centroid is in ROI-local coordinates, map it back to full-image coordinates
            pixel.x = local_centroid.x + bounded.x;
            pixel.y = local_centroid.y + bounded.y;
            return true;
        }

        bool pick_nearest_depth_to_z(const cv::Mat &depth_meters, const cv::Mat &mask, const cv::Rect &roi, float z_ref, float min, float max, cv::Point &pixel)
        {
            if (depth_meters.empty() || mask.empty())
                return false;
            if (!std::isfinite(z_ref) || z_ref <= 0.0f)
                return false;

            float best_delta = std::numeric_limits<float>::infinity();
            cv::Point best_pixel{-1, -1};

            auto func = [&](int x, int y)
            {
                const float depth = depth_meters.at<float>(y, x);
                if (!depth_in_range(depth, min, max))
                    return;
                const float delta = std::fabs(depth - z_ref);
                if (delta < best_delta)
                {
                    best_delta = delta;
                    best_pixel = cv::Point(x, y);
                }
            };

            if (!for_each_mask_pixel_in_roi(mask, roi, func) || best_pixel.x < 0 || best_pixel.y < 0)
                return false;

            pixel = best_pixel;
            return true;
        }

        // Collect depth values from the specified area, optionally applying a mask to filter out invalid pixels.
        // The collected depth values are stored in the output vector.
        bool collect_depth(const cv::Mat &depth_meters, const cv::Rect &sample_area, float min, float max, std::vector<float> &out, const cv::Mat *mask_ptr)
        {
            out.clear();

            cv::Rect bounded;
            if (!clamp_rect_to_image(sample_area, depth_meters.size(), bounded))
                return false;

            out.reserve(static_cast<std::size_t>(bounded.width * bounded.height));
            for (int y = bounded.y; y < bounded.y + bounded.height; ++y)
            {
                const float *drow = depth_meters.ptr<float>(y);
                const unsigned char *mrow = mask_ptr ? mask_ptr->ptr<unsigned char>(y) : nullptr;
                for (int x = bounded.x; x < bounded.x + bounded.width; ++x)
                {
                    if (mrow && mrow[x] == 0)
                        continue;
                    const float d = drow[x];
                    if (depth_in_range(d, min, max))
                        out.push_back(d);
                }
            }

            return !out.empty();
        }

        // Trim the specified ratio of the lowest and highest values from the samples vector. The trim_ratio should be between 0 and 0.49.
        // If the number of samples is less than 3 or the trim_ratio is not positive, no trimming is performed.
        void trim_samples(float trim_ratio, std::vector<float> &samples)
        {
            if (samples.size() < 3 || trim_ratio <= 0.0f)
                return;

            const float ratio = std::max(0.0f, std::min(0.49f, trim_ratio));
            const std::size_t n = samples.size();
            const std::size_t cut = static_cast<std::size_t>(std::floor(static_cast<double>(n) * ratio));
            if (cut == 0 || cut * 2 >= n)
                return;

            auto left = samples.begin() + static_cast<long>(cut);
            auto right = samples.begin() + static_cast<long>(n - cut);
            std::nth_element(samples.begin(), left, samples.end());
            std::nth_element(left, right, samples.end());
            std::move(left, right, samples.begin());
            samples.resize(static_cast<std::size_t>(right - left));
        }
    }

    TargetSelector::TargetSelector(const TargetStrategyConfig &config) : config_(config) {}

    const TargetStrategyConfig &TargetSelector::config() const
    {
        return config_;
    }

    bool TargetSelector::depth_needs_mask() const
    {
        switch (config_.depth_strategy)
        {
        case TargetDepthStrategy::BBoxCenter:
            return false;
        case TargetDepthStrategy::MaskMean:
        case TargetDepthStrategy::MaskMedian:
        case TargetDepthStrategy::MaskedWindowMedian:
        case TargetDepthStrategy::ErodedMaskMedian:
        case TargetDepthStrategy::MaskPercentile:
            return true;
        default:
            return false;
        }
    }

    std::string TargetSelector::depth_strategy_name() const
    {
        switch (config_.depth_strategy)
        {
        case TargetDepthStrategy::BBoxCenter:
            return "BBoxCenter";
        case TargetDepthStrategy::MaskMean:
            return "MaskMean";
        case TargetDepthStrategy::MaskMedian:
            return "MaskMedian";
        case TargetDepthStrategy::MaskedWindowMedian:
            return "MaskedWindowMedian";
        case TargetDepthStrategy::ErodedMaskMedian:
            return "ErodedMaskMedian";
        case TargetDepthStrategy::MaskPercentile:
            return "MaskPercentile";
        default:
            return "Unknown";
        }
    }

    bool TargetSelector::pixel_needs_mask() const
    {
        switch (config_.pixel_strategy)
        {
        case TargetPixelStrategy::BBoxCenter:
            return false;
        case TargetPixelStrategy::MaskCentroid:
        case TargetPixelStrategy::MaskMedian:
        case TargetPixelStrategy::MaskInnerPoint:
        case TargetPixelStrategy::ErodedMaskCentroid:
            return true;
        case TargetPixelStrategy::NearestDepthToZ:
            return depth_needs_mask();
        default:
            return false;
        }
    }

    std::string TargetSelector::pixel_strategy_name() const
    {
        switch (config_.pixel_strategy)
        {
        case TargetPixelStrategy::BBoxCenter:
            return "BBoxCenter";
        case TargetPixelStrategy::MaskCentroid:
            return "MaskCentroid";
        case TargetPixelStrategy::MaskMedian:
            return "MaskMedian";
        case TargetPixelStrategy::MaskInnerPoint:
            return "MaskInnerPoint";
        case TargetPixelStrategy::ErodedMaskCentroid:
            return "ErodedMaskCentroid";
        case TargetPixelStrategy::NearestDepthToZ:
            return "NearestDepthToZ";
        default:
            return "Unknown";
        }
    }

    bool TargetSelector::reduce_depth(std::vector<float> &samples, float &depth_m) const
    {
        depth_m = std::numeric_limits<float>::quiet_NaN();
        if (samples.empty())
            return false;

        if (config_.depth_strategy == TargetDepthStrategy::MaskMean)
        {
            double sum = 0.0;
            for (float v : samples)
                sum += v;
            depth_m = static_cast<float>(sum / static_cast<double>(samples.size()));
            return true;
        }

        if (config_.depth_strategy == TargetDepthStrategy::MaskPercentile)
        {
            const float p = std::max(0.0f, std::min(1.0f, config_.percentile));
            const float pos = p * static_cast<float>(samples.size() - 1);
            const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
            const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));

            std::nth_element(samples.begin(), samples.begin() + static_cast<long>(lo), samples.end());
            const float low = samples[lo];
            if (lo == hi)
            {
                depth_m = low;
                return true;
            }

            std::nth_element(samples.begin(), samples.begin() + static_cast<long>(hi), samples.end());
            const float high = samples[hi];
            depth_m = low + (high - low) * (pos - static_cast<float>(lo));
            return true;
        }
        const std::size_t mid = samples.size() / 2;
        std::nth_element(samples.begin(), samples.begin() + static_cast<long>(mid), samples.end());
        if (samples.size() % 2 == 1)
        {
            depth_m = samples[mid];
            return true;
        }

        const float right = samples[mid];
        std::nth_element(samples.begin(), samples.begin() + static_cast<long>(mid - 1),
                         samples.begin() + static_cast<long>(mid));
        depth_m = 0.5f * (samples[mid - 1] + right);
        return true;
    }

    bool TargetSelector::select(const cv::Mat &depth_meters, const cv::Mat &mask, const cv::Rect &roi) const
    {
        result_ = TargetStrategyResult(); // reset result
        if (!valid_bbox(roi))
        {
            LOG_W("target_strategy", "Invalid ROI for target selection");
            return false;
        }

        if (!is_depth_map_32f(depth_meters))
        {
            LOG_W("target_strategy", "Depth map must be CV_32FC1");
            return false;
        }
        if (!mask.empty() && !is_mask_u8(mask))
        {
            LOG_W("target_strategy", "Mask must be empty or CV_8UC1");
            return false;
        }
        if (!mask.empty() && !mask_matches_depth(depth_meters, mask))
        {
            LOG_W("target_strategy", "Mask/depth size mismatch");
            return false;
        }

        const bool has_mask = !mask.empty();
        auto pixel_strategy = config_.pixel_strategy;
        bool pixel_fallback = false;

        if (!has_mask && pixel_needs_mask())
        {
            pixel_strategy = config_.pixel_fallback_strategy;
            pixel_fallback = true;
        }

        // 像素策略选择
        auto pick_pixel = [&](TargetPixelStrategy strategy, cv::Point &pixel) -> bool
        {
            switch (strategy)
            {
            case TargetPixelStrategy::BBoxCenter:
                return pick_bbox_center(roi, pixel);
            case TargetPixelStrategy::MaskCentroid:
                return pick_mask_centroid(mask, roi, pixel);
            case TargetPixelStrategy::MaskMedian:
                return pick_mask_median(mask, roi, pixel);
            case TargetPixelStrategy::MaskInnerPoint:
                return pick_mask_inner_point(mask, roi, pixel);
            case TargetPixelStrategy::ErodedMaskCentroid:
                return pick_eroded_mask_centroid(mask, roi, config_.erode_kernel, pixel);
            case TargetPixelStrategy::NearestDepthToZ:
            {
                if (!has_mask)
                    return false;

                int window = std::max(1, config_.window_size);
                if (!normalize_odd_kernel(window, window))
                    return false;
                const int radius = window / 2;
                const cv::Point center = roi_center(roi);
                const cv::Rect area(center.x - radius, center.y - radius, window, window);

                std::vector<float> center_samples;
                const cv::Mat *mask_ptr = has_mask ? &mask : nullptr;
                if (!collect_depth(depth_meters, area, config_.min_depth_m, config_.max_depth_m, center_samples, mask_ptr))
                    return false;

                float z_ref = std::numeric_limits<float>::quiet_NaN();
                if (!reduce_depth(center_samples, z_ref))
                    return false;

                return pick_nearest_depth_to_z(depth_meters, mask, roi, z_ref, config_.min_depth_m, config_.max_depth_m, pixel);
            }
            default:
                return false;
            }
        };
        cv::Point pixel(-1, -1);
        if (!pick_pixel(pixel_strategy, pixel))
        {
            if (!pick_pixel(config_.pixel_fallback_strategy, pixel))
            {
                LOG_W("sensor", "Failed to select target pixel");
                return false;
            }
            pixel_fallback = true;
            // LOG_I("sensor", "Pixel strategy '%s' failed, fallback to '%s'", pixel_strategy_name().c_str(), pixel_strategy_name().c_str());
        }
        result_.pixel = pixel;
        result_.used_fallback = pixel_fallback;

        // 深度采样
        auto depth_strategy = config_.depth_strategy;
        bool depth_fallback = false;
        if (!has_mask && depth_needs_mask())
        {
            depth_strategy = TargetDepthStrategy::BBoxCenter;
            depth_fallback = true;
        }

        auto pick_depth = [&](TargetDepthStrategy strategy, float &depth_m) -> bool
        {
            switch (strategy)
            {
            case TargetDepthStrategy::BBoxCenter:
            {
                if (pixel.x < 0 || pixel.y < 0 || pixel.x >= depth_meters.cols || pixel.y >= depth_meters.rows)
                    return false;
                const float d = depth_meters.at<float>(pixel.y, pixel.x);
                if (!depth_in_range(d, config_.min_depth_m, config_.max_depth_m))
                    return false;
                depth_m = d;
                return true;
            }
            case TargetDepthStrategy::MaskMean:
            case TargetDepthStrategy::MaskMedian:
            case TargetDepthStrategy::MaskedWindowMedian:
            case TargetDepthStrategy::ErodedMaskMedian:
            case TargetDepthStrategy::MaskPercentile:
            {
                if (!has_mask)
                    return false;

                std::vector<float> samples;
                bool ok = false;

                if (strategy == TargetDepthStrategy::MaskedWindowMedian)
                {
                    int window = std::max(1, config_.window_size);
                    if (!normalize_odd_kernel(window, window))
                        return false;

                    const int radius = window / 2;
                    const cv::Rect area(pixel.x - radius, pixel.y - radius, window, window);
                    ok = collect_depth(depth_meters, area, config_.min_depth_m, config_.max_depth_m, samples, &mask);
                }
                else if (strategy == TargetDepthStrategy::ErodedMaskMedian)
                {
                    int kernel = 0;
                    if (!normalize_odd_kernel(config_.erode_kernel, kernel))
                        return false;

                    cv::Mat eroded;
                    cv::erode(mask, eroded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel, kernel)));
                    ok = collect_depth(depth_meters, roi, config_.min_depth_m, config_.max_depth_m, samples, &eroded);
                }
                else
                {
                    ok = collect_depth(depth_meters, roi, config_.min_depth_m, config_.max_depth_m, samples, &mask);
                }
                if (!ok)
                    return false;

                trim_samples(config_.trim_ratio, samples);
                return reduce_depth(samples, depth_m);
            }
            default:
                return false;
            }
        };

        float depth_m = std::numeric_limits<float>::quiet_NaN();
        if (!pick_depth(depth_strategy, depth_m))
        {
            const std::string strategy_name = depth_strategy_name();
            LOG_W("target_strategy", "Failed to select target depth by strategy '%s'", strategy_name.c_str());
            return false;
        }
        const std::string strategy_name = depth_strategy_name();
        LOG_I("target_strategy", "Selected target pixel (%d, %d) with depth %.3f m using strategy '%s'%s",
              result_.pixel.x, result_.pixel.y, depth_m, strategy_name.c_str(),
              depth_fallback ? " (fallback)" : "");
        result_.depth_m = depth_m;
        result_.used_fallback |= depth_fallback;
        result_.point3d = cv::Point3d(static_cast<double>(result_.pixel.x), static_cast<double>(result_.pixel.y), static_cast<double>(result_.depth_m));
        result_.valid = std::isfinite(depth_m) && depth_m > 0.0f;
        return result_.valid;
    }
}