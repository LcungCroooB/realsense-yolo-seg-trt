#pragma once

#include <limits>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace sensor
{
    // Depth sampling strategy set for ROI/Mask based object depth estimation.
    enum class TargetDepthStrategy
    {
        BBoxCenter = 0,
        MaskMean,
        MaskMedian,
        MaskedWindowMedian,
        ErodedMaskMedian,
        MaskPercentile
    };

    // Reference pixel selection strategy for depth sampling, used when the depth sampling strategy requires a single pixel reference (e.g., MaskInnerPoint).
    enum class TargetPixelStrategy
    {
        BBoxCenter = 0,
        MaskCentroid,
        MaskMedian,
        MaskInnerPoint,
        ErodedMaskCentroid,
        NearestDepthToZ
    };

    struct TargetStrategyConfig
    {
        float min_depth_m = 0.1f;  // 最小有效深度值
        float max_depth_m = 10.0f; // 最大有效深度值
        int window_size = 3;       // MaskedWindowMedian使用的窗口大小，必须为奇数
        int erode_kernel = 5;      // ErodedMaskMedian使用的腐蚀核大小，必须为奇数
        int min_valid_samples = 5; // 最小有效样本数，低于该数量则认为深度无效
        float trim_ratio = 0.0f;   // 修剪比例，用于去除极端值
        float percentile = 0.3f;   // 百分位数，用于MaskPercentile策略

        TargetDepthStrategy depth_strategy = TargetDepthStrategy::ErodedMaskMedian;
        TargetPixelStrategy pixel_strategy = TargetPixelStrategy::MaskMedian;
        TargetPixelStrategy pixel_fallback_strategy = TargetPixelStrategy::BBoxCenter;
    };

    struct TargetStrategyResult
    {
        bool valid = false;
        bool used_fallback = false;

        float depth_m = std::numeric_limits<float>::quiet_NaN();
        cv::Point pixel{-1, -1};
        cv::Point3d point3d{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    };

    class TargetSelector
    {
    public:
        explicit TargetSelector(const TargetStrategyConfig &config);
        const TargetStrategyConfig &config() const;
        bool select(const cv::Mat &depth_meters, const cv::Mat &mask, const cv::Rect &roi) const;
        
        const TargetStrategyResult &result() const
        {
            return result_;
        }

    private:
        TargetStrategyConfig config_;
        mutable TargetStrategyResult result_;

        bool depth_needs_mask() const;
        std::string depth_strategy_name() const;
        bool pixel_needs_mask() const;
        std::string pixel_strategy_name() const;
        bool reduce_depth(std::vector<float> &samples, float &depth_m) const;
    };
}