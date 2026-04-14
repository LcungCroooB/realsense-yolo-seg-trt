#ifndef __PREPROCESS_KERNEL_CUH__
#define __PREPROCESS_KERNEL_CUH__

#include "cuda_tools.hpp"

namespace trt
{

    enum class NormType : int
    {
        None = 0,
        MeanStd = 1,
        AlphaBeta = 2
    };

    enum class ChannelType : int
    {
        None = 1,
        Invert = 2
    };

    struct Norm
    {
        float mean[3];
        float std[3];
        float alpha, beta;
        NormType type = NormType::None;
        ChannelType channel_type = ChannelType::None;

        // out = (x * alpha - mean) / std  [0 - 1]
        static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f, ChannelType channel_type = ChannelType::None);

        // out = (x * alpha + bata)
        static Norm alpha_bata(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None);

        // Preferred spelling. Kept alongside alpha_bata for compatibility.
        static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None);

        // None
        static Norm None();
    };

    void warp_affine_bilinear_and_normalize_plane(
        uint8_t *src, int src_line_size, int src_width, int src_height,
        float *dst, int dst_width, int dst_height,
        float *matrix_2_3, uint8_t const_value, const Norm &norm,
        cudaStream_t stream);

    void warp_affine_bilinear_and_normalize_focus(
        uint8_t *src, int src_line_size, int src_width, int src_height,
        float *dst, int dst_width, int dst_height,
        float *matrix_2_3, uint8_t const_value, const Norm &norm,
        cudaStream_t stream);

    void resize_bilinear_and_normalize(
        uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
        const Norm &norm,
        cudaStream_t stream);

}; // namespace TRT

#endif