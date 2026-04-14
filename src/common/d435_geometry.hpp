#pragma once

#include <opencv2/opencv.hpp>

#include "d435_sensor.hpp"

namespace sensor
{
    class D435GeometryProjector
    {
    public:
        D435GeometryProjector() = default;
        explicit D435GeometryProjector(const D435CameraParameters &camera_parameters);
        bool set_camera_parameters(const D435CameraParameters &camera_parameters);
        bool valid() const;

        // Back-projection: color pixel + depth(m) -> color camera xyz(m)
        bool deproject_color_pixel(int u, int v, float depth_m, cv::Point3f &point3d) const;
        // Projection: color camera xyz(m) -> color pixel
        bool project_color_point(const cv::Point3f &point3d, cv::Point2f &pixel) const;

        bool transform_depth_to_color(const cv::Point3f &depth_point, cv::Point3f &color_point) const;
        bool transform_color_to_depth(const cv::Point3f &color_point, cv::Point3f &depth_point) const;

    private:
        D435CameraParameters camera_parameters_;
        D435Extrinsics color_to_depth_extrinsics_;
        bool initialized_ = false;
        bool has_transform_ = false;
        };

}