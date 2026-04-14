#include "d435_geometry.hpp"

#include <cmath>
#include <librealsense2/rsutil.h>

namespace sensor
{
    namespace
    {
        bool extrinsics_valid(const D435Extrinsics &extrinsics)
        {
            for (int i = 0; i < 9; ++i)
            {
                if (!std::isfinite(extrinsics.rotation[i]))
                    return false;
            }
            for (int i = 0; i < 3; ++i)
            {
                if (!std::isfinite(extrinsics.translation[i]))
                    return false;
            }
            return true;
        }

        bool intrinsics_valid(const D435Intrinsics &intrinsics)
        {
            return std::isfinite(intrinsics.fx) && std::isfinite(intrinsics.fy) &&
                   std::isfinite(intrinsics.cx) && std::isfinite(intrinsics.cy) &&
                   intrinsics.fx > 0.0f && intrinsics.fy > 0.0f &&
                   intrinsics.width > 0 && intrinsics.height > 0;
        }

        bool finite3(const cv::Point3f &point)
        {
            return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
        }

        rs2_extrinsics to_rs_extrinsics(const D435Extrinsics &extrinsics)
        {
            rs2_extrinsics rs{};
            for (int i = 0; i < 9; ++i)
                rs.rotation[i] = extrinsics.rotation[i];
            for (int i = 0; i < 3; ++i)
                rs.translation[i] = extrinsics.translation[i];
            return rs;
        }

        rs2_intrinsics to_rs_intrinsics(const D435Intrinsics &intrinsics)
        {
            bool has_distortion = false;
            for (int i = 0; i < 5; ++i)
            {
                if (std::fabs(intrinsics.coeffs[i]) > 1e-8f)
                {
                    has_distortion = true;
                    break;
                }
            }

            rs2_intrinsics rs{};
            rs.width = intrinsics.width;
            rs.height = intrinsics.height;
            rs.ppx = intrinsics.cx;
            rs.ppy = intrinsics.cy;
            rs.fx = intrinsics.fx;
            rs.fy = intrinsics.fy;
            rs.model = has_distortion ? RS2_DISTORTION_BROWN_CONRADY : RS2_DISTORTION_NONE;
            for (int i = 0; i < 5; ++i)
                rs.coeffs[i] = intrinsics.coeffs[i];
            return rs;
        }
    }

    D435GeometryProjector::D435GeometryProjector(const D435CameraParameters &camera_parameters)
    {
        set_camera_parameters(camera_parameters);
    }

    bool D435GeometryProjector::set_camera_parameters(const D435CameraParameters &camera_parameters)
    {
        camera_parameters_ = camera_parameters;
        initialized_ = intrinsics_valid(camera_parameters_.color_intrinsics);
        has_transform_ = extrinsics_valid(camera_parameters_.depth_to_color);
        return initialized_;
    }

    bool D435GeometryProjector::valid() const
    {
        return initialized_;
    }

    bool D435GeometryProjector::deproject_color_pixel(int u, int v, float depth_m, cv::Point3f &point3d) const
    {
        if (!initialized_ || !std::isfinite(depth_m) || depth_m <= 0.0f)
            return false;

        const D435Intrinsics &intr = camera_parameters_.color_intrinsics;
        if (u < 0 || v < 0 || u >= intr.width || v >= intr.height)
            return false;

        const rs2_intrinsics rs_intr = to_rs_intrinsics(intr);
        const float pixel[2] = {static_cast<float>(u), static_cast<float>(v)};
        float point[3] = {0.0f, 0.0f, 0.0f};
        rs2_deproject_pixel_to_point(point, &rs_intr, pixel, depth_m);

        point3d = cv::Point3f(point[0], point[1], point[2]);
        return finite3(point3d) && point3d.z > 0.0f;
    }

    bool D435GeometryProjector::project_color_point(const cv::Point3f &point3d, cv::Point2f &pixel) const
    {
        if (!initialized_ || !finite3(point3d) || point3d.z <= 0.0f)
            return false;

        const rs2_intrinsics rs_intr = to_rs_intrinsics(camera_parameters_.color_intrinsics);
        const float in_point[3] = {point3d.x, point3d.y, point3d.z};
        float uv[2] = {0.0f, 0.0f};
        rs2_project_point_to_pixel(uv, &rs_intr, in_point);
        if (!std::isfinite(uv[0]) || !std::isfinite(uv[1]))
            return false;

        pixel = cv::Point2f(uv[0], uv[1]);
        return true;
    }

    bool D435GeometryProjector::transform_depth_to_color(const cv::Point3f &depth_point, cv::Point3f &color_point) const
    {
        if (!initialized_ || !has_transform_ || !finite3(depth_point))
            return false;

        const rs2_extrinsics extr = to_rs_extrinsics(camera_parameters_.depth_to_color);
        const float in_point[3] = {depth_point.x, depth_point.y, depth_point.z};
        float out_point[3] = {0.0f, 0.0f, 0.0f};
        rs2_transform_point_to_point(out_point, &extr, in_point);

        color_point = cv::Point3f(out_point[0], out_point[1], out_point[2]);
        return finite3(color_point);
    }

    bool D435GeometryProjector::transform_color_to_depth(const cv::Point3f &color_point, cv::Point3f &depth_point) const
    {
        if (!initialized_ || !has_transform_ || !finite3(color_point))
            return false;

        const rs2_extrinsics extr = to_rs_extrinsics(color_to_depth_extrinsics_);
        const float in_point[3] = {color_point.x, color_point.y, color_point.z};
        float out_point[3] = {0.0f, 0.0f, 0.0f};
        rs2_transform_point_to_point(out_point, &extr, in_point);

        depth_point = cv::Point3f(out_point[0], out_point[1], out_point[2]);
        return finite3(depth_point);
    }

}