#include "GaussianBlur.hpp"
#include <algorithm>
#include <cmath>

namespace Gaussian
{
    std::vector<std::vector<float>> generateGaussianKernel(int radius, float sigma)
    {
        int size = 2 * radius + 1;
        std::vector<std::vector<float>> kernel(size, std::vector<float>(size));

        float sum = 0.0f;
        float invTwoSigma2 = 1.0f / (2.0f * sigma * sigma);

        for (int y = -radius; y <= radius; ++y)
        {
            for (int x = -radius; x <= radius; ++x)
            {
                float value = std::exp(-(x*x + y*y) * invTwoSigma2);
                kernel[y + radius][x + radius] = value;
                sum += value;
            }
        }

        // Normalize so the kernel sums to 1
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
                kernel[y][x] /= sum;

        return kernel;
    }

    std::vector<float> gaussianBlur2D_openMP(
        const std::vector<float>& input,
        int width,
        int height,
        const std::vector<std::vector<float>>& kernel)
    {
        int radius = kernel.size() / 2;
        std::vector<float> output(width * height, 0.0f);

        #pragma omp parallel for schedule(static)
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float sum = 0.0f;

                for (int ky = -radius; ky <= radius; ++ky)
                {
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        int iy = std::clamp(y + ky, 0, height - 1);
                        int ix = std::clamp(x + kx, 0, width - 1);

                        float pixel = input[iy * width + ix];
                        float weight = kernel[ky + radius][kx + radius];

                        sum += pixel * weight;
                    }
                }

                output[y * width + x] = sum;
            }
        }

        return output;
    }
}