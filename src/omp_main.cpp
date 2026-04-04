/**
 * Running omp_main.cpp requires the program to know.
 * 1. How many threads you want to run (argv[1]: nThreads)
 * 2. What image you want to process (argv[2]: imgName)
 */

#include <iostream>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
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

    //     pixelComponent=#comp     components
    //       1                          grey
    //       2                          grey, alpha
    //       3                          red, green, blue
    //       4                          red, green, blue, alpha

    extractedPixels = stbi_load(image, &imgWidth, &imgHeight, &pixelComponent, 4);

    if (extractedPixels == NULL)
    {
        std::cout << "File was not found. Please try again." << std::endl;
        exit(1);
    }

    //}
    // catch (Exception ex)
    // {
    // }
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

#pragma omp critical
        {
            std::cout << "Hello from thread " << thread_id << " of " << total_threads << std::endl;
        }
    }

    return 0;
}
