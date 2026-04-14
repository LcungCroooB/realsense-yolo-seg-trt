#pragma once

#include <cmath>
#include <future>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

#include "benchmark_common.hpp"
#include "../common/utils.hpp"
#include "../logger/logger_macro.h"
#include "../tasks/yolo.hpp"
#include "../tasks/yolo_obb.hpp"
#include "../tasks/yolo_seg.hpp"
#include "../tasks/yolo_pose.hpp"

namespace app_benchmark
{
    namespace traits
    {
        // =====================================================================
        // DET Task Traits
        // =====================================================================
        class DetTraits
        {
        public:
            static constexpr const char *task_name = "DET";
            using YoloTypeEnum = yolo::Type;
            using InferType = std::shared_ptr<yolo::Infer>;
            using BoxArrayType = yolo::BoxArray;

            static bool create_infer(
                const std::string &engine_file,
                const std::string &yolo_type_str,
                int gpu_id,
                InferType &infer)
            {
                const yolo::Type type = yolo::type_from_string(yolo_type_str);
                infer = yolo::create_infer(engine_file, type, gpu_id, 0.25f, 0.45f,
                                           yolo::NMSMethod::FastGPU, 1024, false);
                return infer != nullptr;
            }

            static void run_single_inference(InferType &infer, const cv::Mat &image)
            {
                infer->commit(image).get();
            }

            static void run_batch_inference(InferType &infer, const std::vector<cv::Mat> &images)
            {
                std::vector<std::shared_future<BoxArrayType>> futures = infer->commits(images);
                for (std::shared_future<BoxArrayType> &f : futures)
                    f.get();
            }

            static BoxArrayType get_results(InferType &infer, const cv::Mat &image)
            {
                return infer->commit(image).get();
            }

            static yolo::Type get_yolo_type(const std::string &text)
            {
                return yolo::type_from_string(text);
            }

            static const char *type_name_fn(yolo::Type type)
            {
                return yolo::type_name(type);
            }

            static void save_visualizations(
                const std::string &result_dir,
                int save_count,
                const std::vector<LoadedImage> &images,
                InferType &infer)
            {
                if (save_count <= 0 || images.empty())
                    return;

                if (!utils::fs::mkdirs(result_dir))
                {
                    LOG_E("bench", "Failed to create result directory: %s", result_dir.c_str());
                    return;
                }

                const long long stamp = static_cast<long long>(utils::time::timestamp_millisecond());
                const int count = std::min(static_cast<int>(images.size()), save_count);
                for (int i = 0; i < count; ++i)
                {
                    const auto &item = images[static_cast<size_t>(i)];
                    cv::Mat canvas = item.image.clone();
                    const BoxArrayType boxes = get_results(infer, item.image);
                    draw_det_boxes(canvas, boxes);

                    const std::string output_path = utils::format(
                        "%s/vis_%lld_%02d_%s.jpg",
                        result_dir.c_str(), stamp, i,
                        utils::fs::file_name(item.path, false).c_str());

                    if (!cv::imwrite(output_path, canvas))
                    {
                        LOG_E("bench", "Failed to save visualization: %s", output_path.c_str());
                        continue;
                    }
                    LOG_I("bench", "Saved visualization to %s", output_path.c_str());
                }
            }

        private:
            static void draw_det_boxes(cv::Mat &image, const yolo::BoxArray &boxes)
            {
                for (const auto &box : boxes)
                {
                    const auto color = utils::random_color(box.class_label);
                    const cv::Scalar bgr(
                        static_cast<int>(std::get<0>(color)),
                        static_cast<int>(std::get<1>(color)),
                        static_cast<int>(std::get<2>(color)));

                    const int left = clamp_to_int(box.left, 0, std::max(0, image.cols - 1));
                    const int top = clamp_to_int(box.top, 0, std::max(0, image.rows - 1));
                    const int right = clamp_to_int(box.right, 0, std::max(0, image.cols - 1));
                    const int bottom = clamp_to_int(box.bottom, 0, std::max(0, image.rows - 1));

                    cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), bgr, 2, cv::LINE_AA);
                }
            }
        };

        // =====================================================================
        // OBB Task Traits
        // =====================================================================
        class ObbTraits
        {
        public:
            static constexpr const char *task_name = "OBB";
            using YoloTypeEnum = yoloobb::Type;
            using InferType = std::shared_ptr<yoloobb::Infer>;
            using BoxArrayType = yoloobb::BoxArray;

            static bool create_infer(
                const std::string &engine_file,
                const std::string &yolo_type_str,
                int gpu_id,
                InferType &infer)
            {
                const yoloobb::Type type = yoloobb::type_from_string(yolo_type_str);
                infer = yoloobb::create_infer(engine_file, type, gpu_id, 0.25f, 0.45f,
                                              yoloobb::NMSMethod::FastGPU, 1024, false);
                return infer != nullptr;
            }

            static void run_single_inference(InferType &infer, const cv::Mat &image)
            {
                infer->commit(image).get();
            }

            static void run_batch_inference(InferType &infer, const std::vector<cv::Mat> &images)
            {
                std::vector<std::shared_future<BoxArrayType>> futures = infer->commits(images);
                for (std::shared_future<BoxArrayType> &f : futures)
                    f.get();
            }

            static BoxArrayType get_results(InferType &infer, const cv::Mat &image)
            {
                return infer->commit(image).get();
            }

            static yoloobb::Type get_yolo_type(const std::string &text)
            {
                return yoloobb::type_from_string(text);
            }

            static const char *type_name_fn(yoloobb::Type type)
            {
                return yoloobb::type_name(type);
            }

            static void save_visualizations(
                const std::string &result_dir,
                int save_count,
                const std::vector<LoadedImage> &images,
                InferType &infer)
            {
                if (save_count <= 0 || images.empty())
                    return;

                if (!utils::fs::mkdirs(result_dir))
                {
                    LOG_E("bench", "Failed to create result directory: %s", result_dir.c_str());
                    return;
                }

                const long long stamp = static_cast<long long>(utils::time::timestamp_millisecond());
                const int count = std::min(static_cast<int>(images.size()), save_count);
                for (int i = 0; i < count; ++i)
                {
                    const auto &item = images[static_cast<size_t>(i)];
                    cv::Mat canvas = item.image.clone();
                    const BoxArrayType boxes = get_results(infer, item.image);
                    draw_obb_boxes(canvas, boxes);

                    const std::string output_path = utils::format(
                        "%s/vis_obb_%lld_%02d_%s.jpg",
                        result_dir.c_str(), stamp, i,
                        utils::fs::file_name(item.path, false).c_str());

                    if (!cv::imwrite(output_path, canvas))
                    {
                        LOG_E("bench", "Failed to save OBB visualization: %s", output_path.c_str());
                        continue;
                    }
                    LOG_I("bench", "Saved OBB visualization to %s", output_path.c_str());
                }
            }

        private:
            static std::vector<cv::Point> xywhr2xyxyxyxy(const yoloobb::Box &box, const cv::Mat &image)
            {
                const float cos_value = std::cos(box.angle);
                const float sin_value = std::sin(box.angle);

                const float half_width = box.width * 0.5f;
                const float half_height = box.height * 0.5f;
                const float vec1_x = half_width * cos_value;
                const float vec1_y = half_width * sin_value;
                const float vec2_x = -half_height * sin_value;
                const float vec2_y = half_height * cos_value;

                std::vector<cv::Point> corners;
                corners.reserve(4);
                corners.emplace_back(
                    clamp_to_int(box.center_x + vec1_x + vec2_x, 0, std::max(0, image.cols - 1)),
                    clamp_to_int(box.center_y + vec1_y + vec2_y, 0, std::max(0, image.rows - 1)));
                corners.emplace_back(
                    clamp_to_int(box.center_x + vec1_x - vec2_x, 0, std::max(0, image.cols - 1)),
                    clamp_to_int(box.center_y + vec1_y - vec2_y, 0, std::max(0, image.rows - 1)));
                corners.emplace_back(
                    clamp_to_int(box.center_x - vec1_x - vec2_x, 0, std::max(0, image.cols - 1)),
                    clamp_to_int(box.center_y - vec1_y - vec2_y, 0, std::max(0, image.rows - 1)));
                corners.emplace_back(
                    clamp_to_int(box.center_x - vec1_x + vec2_x, 0, std::max(0, image.cols - 1)),
                    clamp_to_int(box.center_y - vec1_y + vec2_y, 0, std::max(0, image.rows - 1)));
                return corners;
            }

            static void draw_obb_boxes(cv::Mat &image, const yoloobb::BoxArray &boxes)
            {
                for (const auto &box : boxes)
                {
                    const auto color = utils::random_color(box.class_label);
                    const cv::Scalar bgr(
                        static_cast<int>(std::get<0>(color)),
                        static_cast<int>(std::get<1>(color)),
                        static_cast<int>(std::get<2>(color)));

                    const double angle = box.angle * 180.0 / 3.14159265;
                    const std::vector<cv::Point> corners = xywhr2xyxyxyxy(box, image);

                    cv::polylines(image, std::vector<std::vector<cv::Point>>{corners}, true, bgr, 2, cv::LINE_AA);
                    cv::circle(
                        image,
                        cv::Point(
                            clamp_to_int(box.center_x, 0, std::max(0, image.cols - 1)),
                            clamp_to_int(box.center_y, 0, std::max(0, image.rows - 1))),
                        3,
                        bgr,
                        -1,
                        cv::LINE_AA);
                }
            }
        };

        // =====================================================================
        // SEG Task Traits
        // =====================================================================
        class SegTraits
        {
        public:
            static constexpr const char *task_name = "SEG";
            using YoloTypeEnum = yoloseg::Type;
            using InferType = std::shared_ptr<yoloseg::Infer>;
            using BoxArrayType = yoloseg::BoxArray;

            static bool create_infer(
                const std::string &engine_file,
                const std::string &yolo_type_str,
                int gpu_id,
                InferType &infer)
            {
                const yoloseg::Type type = yoloseg::type_from_string(yolo_type_str);
                infer = yoloseg::create_infer(engine_file, type, gpu_id, 0.25f, 0.45f,
                                              yoloseg::NMSMethod::FastGPU, 1024, false);
                return infer != nullptr;
            }

            static void run_single_inference(InferType &infer, const cv::Mat &image)
            {
                infer->commit(image).get();
            }

            static void run_batch_inference(InferType &infer, const std::vector<cv::Mat> &images)
            {
                std::vector<std::shared_future<BoxArrayType>> futures = infer->commits(images);
                for (std::shared_future<BoxArrayType> &f : futures)
                    f.get();
            }

            static BoxArrayType get_results(InferType &infer, const cv::Mat &image)
            {
                return infer->commit(image).get();
            }

            static yoloseg::Type get_yolo_type(const std::string &text)
            {
                return yoloseg::type_from_string(text);
            }

            static const char *type_name_fn(yoloseg::Type type)
            {
                return yoloseg::type_name(type);
            }

            static void save_visualizations(
                const std::string &result_dir,
                int save_count,
                const std::vector<LoadedImage> &images,
                InferType &infer)
            {
                if (save_count <= 0 || images.empty())
                    return;

                if (!utils::fs::mkdirs(result_dir))
                {
                    LOG_E("bench", "Failed to create result directory: %s", result_dir.c_str());
                    return;
                }

                const long long stamp = static_cast<long long>(utils::time::timestamp_millisecond());
                const int count = std::min(static_cast<int>(images.size()), save_count);
                for (int i = 0; i < count; ++i)
                {
                    const auto &item = images[static_cast<size_t>(i)];
                    cv::Mat canvas = item.image.clone();
                    const BoxArrayType boxes = get_results(infer, item.image);
                    draw_seg_boxes(canvas, boxes);

                    const std::string output_path = utils::format(
                        "%s/vis_seg_%lld_%02d_%s.jpg",
                        result_dir.c_str(), stamp, i,
                        utils::fs::file_name(item.path, false).c_str());

                    if (!cv::imwrite(output_path, canvas))
                    {
                        LOG_E("bench", "Failed to save SEG visualization: %s", output_path.c_str());
                        continue;
                    }
                    LOG_I("bench", "Saved SEG visualization to %s", output_path.c_str());
                }
            }

        private:
            static void draw_mask(
                cv::Mat &image,
                const yoloseg::Box &box,
                const cv::Scalar &color,
                int proto_width,
                int proto_height)
            {
                if (!box.seg || proto_width <= 0 || proto_height <= 0 || image.empty())
                    return;

                const int input_width = proto_width * 4;
                const int input_height = proto_height * 4;
                const float scale_x = static_cast<float>(input_width) / static_cast<float>(image.cols);
                const float scale_y = static_cast<float>(input_height) / static_cast<float>(image.rows);
                const float scale = std::min(scale_x, scale_y);
                const float ox = -scale * image.cols * 0.5f + input_width * 0.5f + scale * 0.5f - 0.5f;
                const float oy = -scale * image.rows * 0.5f + input_height * 0.5f + scale * 0.5f - 0.5f;

                const cv::Mat affine = (cv::Mat_<float>(2, 3) << scale, 0, ox, 0, scale, oy);
                cv::Mat inverse_affine;
                cv::invertAffineTransform(affine, inverse_affine);

                cv::Mat mask_map = cv::Mat::zeros(cv::Size(proto_width, proto_height), CV_8UC1);
                cv::Mat small_mask(box.seg->height, box.seg->width, CV_8UC1, box.seg->data);

                const int roi_x = std::max(0, std::min(box.seg->left, proto_width - 1));
                const int roi_y = std::max(0, std::min(box.seg->top, proto_height - 1));
                const int roi_w = std::min(box.seg->width, proto_width - roi_x);
                const int roi_h = std::min(box.seg->height, proto_height - roi_y);
                if (roi_w <= 0 || roi_h <= 0)
                    return;

                small_mask(cv::Rect(0, 0, roi_w, roi_h)).copyTo(mask_map(cv::Rect(roi_x, roi_y, roi_w, roi_h)));

                cv::resize(mask_map, mask_map, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);
                cv::threshold(mask_map, mask_map, 128, 1, cv::THRESH_BINARY);

                cv::Mat mask_resized;
                cv::warpAffine(mask_map, mask_resized, inverse_affine, image.size(), cv::INTER_LINEAR);

                cv::Mat colored_mask = cv::Mat::ones(image.size(), CV_8UC3);
                colored_mask.setTo(color);

                cv::Mat masked_colored_mask;
                cv::bitwise_and(colored_mask, colored_mask, masked_colored_mask, mask_resized);

                cv::Mat mask_indices;
                cv::compare(mask_resized, 1, mask_indices, cv::CMP_EQ);

                cv::Mat image_masked;
                cv::Mat colored_mask_masked;
                image.copyTo(image_masked, mask_indices);
                masked_colored_mask.copyTo(colored_mask_masked, mask_indices);

                cv::Mat result_masked;
                cv::addWeighted(image_masked, 0.6, colored_mask_masked, 0.4, 0, result_masked);
                result_masked.copyTo(image, mask_indices);
            }

            static void draw_seg_boxes(cv::Mat &image, const yoloseg::BoxArray &boxes)
            {
                int proto_width = 160;
                int proto_height = 160;
                for (const auto &box : boxes)
                {
                    if (!box.seg)
                        continue;
                    proto_width = std::max(proto_width, box.seg->left + box.seg->width);
                    proto_height = std::max(proto_height, box.seg->top + box.seg->height);
                }

                for (const auto &box : boxes)
                {
                    if (!box.seg)
                        continue;
                    const auto color = utils::random_color(box.class_label);
                    const cv::Scalar bgr(
                        static_cast<int>(std::get<0>(color)),
                        static_cast<int>(std::get<1>(color)),
                        static_cast<int>(std::get<2>(color)));
                    draw_mask(image, box, bgr, proto_width, proto_height);
                }

                for (const auto &box : boxes)
                {
                    const auto color = utils::random_color(box.class_label);
                    const cv::Scalar bgr(
                        static_cast<int>(std::get<0>(color)),
                        static_cast<int>(std::get<1>(color)),
                        static_cast<int>(std::get<2>(color)));

                    const int left = clamp_to_int(box.left, 0, std::max(0, image.cols - 1));
                    const int top = clamp_to_int(box.top, 0, std::max(0, image.rows - 1));
                    const int right = clamp_to_int(box.right, 0, std::max(0, image.cols - 1));
                    const int bottom = clamp_to_int(box.bottom, 0, std::max(0, image.rows - 1));

                    cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), bgr, 5, cv::LINE_AA);
                }
            }
        };

        // =====================================================================
        // POSE Task Traits
        // =====================================================================
        class PoseTraits
        {
        public:
            static constexpr const char *task_name = "POSE";
            using YoloTypeEnum = yolopose::Type;
            using InferType = std::shared_ptr<yolopose::Infer>;
            using BoxArrayType = yolopose::BoxArray;

            static bool create_infer(
                const std::string &engine_file,
                const std::string &yolo_type_str,
                int gpu_id,
                InferType &infer)
            {
                const yolopose::Type type = yolopose::type_from_string(yolo_type_str);
                infer = yolopose::create_infer(engine_file, type, gpu_id, 0.25f, 0.45f,
                                               yolopose::NMSMethod::FastGPU, 1024, false);
                return infer != nullptr;
            }

            static void run_single_inference(InferType &infer, const cv::Mat &image)
            {
                infer->commit(image).get();
            }

            static void run_batch_inference(InferType &infer, const std::vector<cv::Mat> &images)
            {
                std::vector<std::shared_future<BoxArrayType>> futures = infer->commits(images);
                for (std::shared_future<BoxArrayType> &f : futures)
                    f.get();
            }

            static BoxArrayType get_results(InferType &infer, const cv::Mat &image)
            {
                return infer->commit(image).get();
            }

            static yolopose::Type get_yolo_type(const std::string &text)
            {
                return yolopose::type_from_string(text);
            }

            static const char *type_name_fn(yolopose::Type type)
            {
                return yolopose::type_name(type);
            }

            static void save_visualizations(
                const std::string &result_dir,
                int save_count,
                const std::vector<LoadedImage> &images,
                InferType &infer)
            {
                if (save_count <= 0 || images.empty())
                    return;

                if (!utils::fs::mkdirs(result_dir))
                {
                    LOG_E("bench", "Failed to create result directory: %s", result_dir.c_str());
                    return;
                }

                const long long stamp = static_cast<long long>(utils::time::timestamp_millisecond());
                const int count = std::min(static_cast<int>(images.size()), save_count);
                for (int i = 0; i < count; ++i)
                {
                    const auto &item = images[static_cast<size_t>(i)];
                    cv::Mat canvas = item.image.clone();
                    const BoxArrayType boxes = get_results(infer, item.image);
                    draw_pose_keypoints(canvas, boxes);

                    const std::string output_path = utils::format(
                        "%s/vis_pose_%lld_%02d_%s.jpg",
                        result_dir.c_str(), stamp, i,
                        utils::fs::file_name(item.path, false).c_str());

                    if (!cv::imwrite(output_path, canvas))
                    {
                        LOG_E("bench", "Failed to save POSE visualization: %s", output_path.c_str());
                        continue;
                    }
                    LOG_I("bench", "Saved POSE visualization to %s", output_path.c_str());
                }
            }

        private:
            static void draw_pose(cv::Mat &image, const std::vector<cv::Point3f> &keypoints)
            {
                static const std::vector<cv::Scalar> pose_palette = {
                    {255, 128, 0}, {255, 153, 51}, {255, 178, 102}, {230, 230, 0}, {255, 153, 255}, {153, 204, 255}, {255, 102, 255}, {255, 51, 255}, {102, 178, 255}, {51, 153, 255}, {255, 153, 153}, {255, 102, 102}, {255, 51, 51}, {153, 255, 153}, {102, 255, 102}, {51, 255, 51}, {0, 255, 0}, {0, 0, 255}, {255, 0, 0}, {255, 255, 255}};

                static const std::vector<cv::Point> skeleton = {
                    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}};

                static const std::vector<cv::Scalar> limb_color = {
                    pose_palette[9], pose_palette[9], pose_palette[9], pose_palette[9], pose_palette[7],
                    pose_palette[7], pose_palette[7], pose_palette[0], pose_palette[0], pose_palette[0],
                    pose_palette[0], pose_palette[0], pose_palette[16], pose_palette[16], pose_palette[16],
                    pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16]};

                static const std::vector<cv::Scalar> kpt_color = {
                    pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16],
                    pose_palette[0], pose_palette[0], pose_palette[0], pose_palette[0], pose_palette[0],
                    pose_palette[0], pose_palette[9], pose_palette[9], pose_palette[9], pose_palette[9],
                    pose_palette[9], pose_palette[9]};

                for (size_t i = 0; i < keypoints.size() && i < kpt_color.size(); ++i)
                {
                    const cv::Point3f &keypoint = keypoints[i];
                    if (keypoint.z < 0.5f)
                        continue;
                    if (keypoint.x != 0.0f && keypoint.y != 0.0f)
                    {
                        cv::circle(
                            image,
                            cv::Point(
                                clamp_to_int(keypoint.x, 0, std::max(0, image.cols - 1)),
                                clamp_to_int(keypoint.y, 0, std::max(0, image.rows - 1))),
                            5,
                            kpt_color[i],
                            -1,
                            cv::LINE_AA);
                    }
                }

                for (size_t i = 0; i < skeleton.size() && i < limb_color.size(); ++i)
                {
                    const cv::Point &index = skeleton[i];
                    if (index.x >= static_cast<int>(keypoints.size()) || index.y >= static_cast<int>(keypoints.size()))
                        continue;

                    const cv::Point3f &pos1 = keypoints[index.x];
                    const cv::Point3f &pos2 = keypoints[index.y];
                    if (pos1.z < 0.5f || pos2.z < 0.5f)
                        continue;
                    if (pos1.x == 0.0f || pos1.y == 0.0f || pos2.x == 0.0f || pos2.y == 0.0f)
                        continue;

                    cv::line(
                        image,
                        cv::Point(
                            clamp_to_int(pos1.x, 0, std::max(0, image.cols - 1)),
                            clamp_to_int(pos1.y, 0, std::max(0, image.rows - 1))),
                        cv::Point(
                            clamp_to_int(pos2.x, 0, std::max(0, image.cols - 1)),
                            clamp_to_int(pos2.y, 0, std::max(0, image.rows - 1))),
                        limb_color[i],
                        2,
                        cv::LINE_AA);
                }
            }

            static void draw_pose_keypoints(cv::Mat &image, const yolopose::BoxArray &boxes)
            {
                for (const auto &box : boxes)
                {
                    const auto color = utils::random_color(box.class_label);
                    const cv::Scalar bgr(
                        static_cast<int>(std::get<0>(color)),
                        static_cast<int>(std::get<1>(color)),
                        static_cast<int>(std::get<2>(color)));

                    const int left = clamp_to_int(box.left, 0, std::max(0, image.cols - 1));
                    const int top = clamp_to_int(box.top, 0, std::max(0, image.rows - 1));
                    const int right = clamp_to_int(box.right, 0, std::max(0, image.cols - 1));
                    const int bottom = clamp_to_int(box.bottom, 0, std::max(0, image.rows - 1));

                    cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), bgr, 2, cv::LINE_AA);

                    draw_pose(image, box.keypoints);
                }
            }
        };
    }
}
