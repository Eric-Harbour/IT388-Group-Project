/**
 * Running omp_main.cpp requires the program to know.
 * 1. How many threads you want to run (argv[1]: nThreads)
 * 2. What image you want to process (argv[2]: imgName)
 * 3. What we want our radius to be
 */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <omp.h>
#include "stb_image.h"
// #include "stb_image_write.h"
#include "GaussianBlur.hpp"

// #include <string.h>

int main(int argc, char **argv)
{
    /*Check user input*/
    if (argc != 3)
    {
        std::cout << "Usage: ./omp_main <number-of-threads> <image-file-name>" << std::endl;
    }

    char *image = argv[2];
    unsigned char *extractedPixels;
    int imgWidth, imgHeight, pixelComponent;
    int radius;

    //     pixelComponent=#comp     components
    //       1                          grey
    //       2                          grey, alpha
    //       3                          red, green, blue
    //       4                          red, green, blue, alpha

    extractedPixels = stbi_load(image, &imgWidth, &imgHeight, &pixelComponent, 4);

    if (extractedPixels == NULL)
    {
        std::cout << "Failed to load image: " << stbi_failure_reason() << std::endl;
        exit(1);
    }

    /**
     * Create our Gaussian Blur object
     */

    /*We will get our radius from user input*/

    /* convert unsigned char to vector*/
    std::vector<float> pixelVector;
    unsigned char r, g, b;

    for (int i = 0; i < imgWidth * imgHeight * pixelComponent; i += pixelComponent)
    {
        r = extractedPixels[i];
        g = extractedPixels[i + 1];
        b = extractedPixels[i + 2];
        pixelVector.push_back((float)r);
        pixelVector.push_back((float)g);
        pixelVector.push_back((float)b);
    }

    /**
     * Prints Pixels
     */
    // for (int i = 0; i < imgWidth * imgHeight * pixelComponent; i += pixelComponent)
    // {
    //     r = pixelVector[i];
    //     g = pixelVector[i + 1];
    //     b = pixelVector[i + 2];
    //     std::cout << "Pixel # " << i << ":" << std::endl;
    //     std::cout << "[" << (float)r * .01 << "," << (float)g << "," << (float)b << "]" << std::endl;
    // }
    float sigma = 1 / imgWidth;
    radius = 3;
    std::vector<std::vector<float>> gaussianKernel = Gaussian::generateGaussianKernel(radius, sigma);

    pixelVector = Gaussian::gaussianBlur2D_openMP(pixelVector, imgWidth, imgHeight, gaussianKernel);

    for (int i = 0; i < imgWidth * imgHeight * pixelComponent; i += pixelComponent)
    {
        r = pixelVector[i];
        g = pixelVector[i + 1];
        b = pixelVector[i + 2];
        std::cout << "Pixel # " << i << ":" << std::endl;
        std::cout << "[" << (float)r * .01 << "," << (float)g << "," << (float)b << "]" << std::endl;
    }

    // #pragma omp parallel
    //     {
    //         // int thread_id = omp_get_thread_num();
    //         // int total_threads = omp_get_num_threads();

    // #pragma omp critical
    //         {
    //             std::cout << "Hello from thread " << thread_id << " of " << total_threads << std::endl;
    //         }
    //     }

    return 0;
}
