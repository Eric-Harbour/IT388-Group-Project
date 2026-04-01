#ifndef GAUSSIAN_BLUR_HPP
#define GAUSSIAN_BLUR_HPP

#include <vector>

namespace Gaussian
{
    // Generate a 2D Gaussian kernel
    std::vector<std::vector<float>> generateGaussianKernel(int radius, float sigma);

    // Apply a 2D Gaussian blur (full convolution)
    std::vector<float> gaussianBlur2D(
        const std::vector<float>& input,
        int width,
        int height,
        const std::vector<std::vector<float>>& kernel
    );
}

#endif