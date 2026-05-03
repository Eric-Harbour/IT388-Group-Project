/* File: cuda_main.c
 * Name: Garrett Blankenship, Jeffrey Armour, Eric Harbour, and Nathan Hilbert 
 * Course: IT 388 - Spring 2026
 * Assignment: Gaussian Blur Project
 */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cassert>
#include <iostream>
#include <utility>

// Simple struct to keep track of image metadata and the raw pixel data
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::byte* data = nullptr;
    bool transposed = false;
};

// Loads an image from disk and moves it into CUDA managed memory
Image create_image(const std::string& file) {
    Image image;

    // Force 4 channels (RGBA) for simplicity
    auto* rawData = reinterpret_cast<std::byte*>(stbi_load(file.c_str(), &image.width, &image.height, &image.channels, 4));
    image.channels = 4; // Force the image to be 4 channels in the meta data because req_comp is 4

    if (!rawData) {
        std::cerr << "Failed to load image" << std::endl;
        assert(rawData != nullptr && "Failed to load image");
    }

    // Use managed memory so both CPU and GPU can access it easily
    cudaMallocManaged(&image.data, image.width * image.height * 4);
    cudaMemcpy(image.data, rawData, image.width * image.height * 4, cudaMemcpyHostToDevice);
    stbi_image_free(rawData);

    return image;
}

Image create_empty_image(unsigned int width, unsigned int height) {
    Image image;
    image.width = width;
    image.height = height;
    image.channels = 4;
    cudaMallocManaged(&image.data, width * height * 4);
    cudaMemset(image.data, 0, width * height * image.channels);
    return image;
}

void free_image(Image& image) {
    if (image.data) cudaFree(image.data);
    image.data = nullptr;
    image.width = 0;
    image.height = 0;
    image.channels = 0;
    image.transposed = false;
}

// Swaps rows and columns. This lets us use the same horizontal blur kernel for vertical blurring.
void transpose(Image& image) {
    if (!image.data) return;

    std::byte* newData;
    cudaMallocManaged(&newData, image.width * image.height * image.channels);

    for (int i = 0; i < image.width; i++) {
        for (int j = 0; j < image.height; j++) {
            for (int k = 0; k < image.channels; k++) {
                newData[(j + i * image.height) * image.channels + k] = image.data[(i + j * image.width) * image.channels + k];
            }
        }
    }

    cudaFree(image.data);
    image.data = newData;
    std::swap(image.width, image.height);
    image.transposed = !image.transposed;
}

void save_image(Image& image, const std::string& file) {
    if (!image.data)
        return;

    stbi_write_png(file.c_str(), image.width, image.height, image.channels, image.data, 0);
}

// 1D Gaussian blur kernel. We run this twice (with a transpose in between) to get a full 2D blur.
__global__ void horizontal_blur(const Image& input, Image& output, float sigma, int radius) {
    assert(input.width == output.width && input.height == output.height && input.channels == output.channels);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= input.width || y >= input.height)
        return;

    float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;

    // Apply the Gaussian weights to the surrounding pixels in the current row
    for (int i = -radius; i <= radius; i++) {
        int nx = x + i;
        if (nx >= 0 && nx < input.width) {
            int idx = (y * input.width + nx) * 4;
            float weight = expf(-(i * i) / (2.0f * sigma * sigma));

            color.x += (unsigned char)input.data[idx] * weight;
            color.y += (unsigned char)input.data[idx + 1] * weight;
            color.z += (unsigned char)input.data[idx + 2] * weight;
            color.w += (unsigned char)input.data[idx + 3] * weight;
            totalWeight += weight;
        }
    }

    int outIdx = (y * input.width + x) * 4;
    output.data[outIdx] = (std::byte)(color.x / totalWeight);
    output.data[outIdx + 1] = (std::byte)(color.y / totalWeight);
    output.data[outIdx + 2] = (std::byte)(color.z / totalWeight);
    output.data[outIdx + 3] = (std::byte)(color.w / totalWeight);
}

int main(int argc, char** argv) {
    std::string outputPath = "./resources/output.png";
    float sigma = 1.0f;
    int radius = 2;
    int threadCountRow = 16;

    // Initialize the arguments
    if (argc < 2) {
        std::printf("Usage: %s (image-file-name) [output-file-name] [sigma] [radius]\n", argv[0]);
        return 1;
    }

    if (argc > 2) outputPath = argv[2];
    if (argc > 3) sigma = std::stof(argv[3]);
    if (argc > 4) radius = std::stoi(argv[4]);
    if (argc > 5) threadCountRow = std::stoi(argv[5]);

    std::printf("Running blur on %s with sigma %f and radius %d with CUDA. Using %d threads.", argv[1], sigma, radius, threadCountRow * threadCountRow);

    // Extract pixels with pixelComponent=4 (red, green, blue, alpha)
    Image inputImage = create_image(argv[1]);
    Image outputImage = create_empty_image(inputImage.width, inputImage.height);

    dim3 blockDim(threadCountRow, threadCountRow);
    dim3 gridDim((inputImage.width + blockDim.x - 1) / blockDim.x, (inputImage.height + blockDim.y - 1) / blockDim.y);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Pass 1: Horizontal Blur
    horizontal_blur<<<gridDim, blockDim>>>(inputImage, outputImage, sigma, radius);
    cudaDeviceSynchronize();

    // To do the vertical blur, we flip the image and run the horizontal kernel again.
    // This is often faster than writing a separate vertical kernel because of memory access patterns.
    transpose(outputImage);
    std::swap(inputImage.width, inputImage.height);

    // Swap inputImage dimensions to match transposed outputImage for the blur
    gridDim = dim3((outputImage.width + blockDim.x - 1) / blockDim.x, (outputImage.height + blockDim.y - 1) / blockDim.y);
    horizontal_blur<<<gridDim, blockDim>>>(outputImage, inputImage, sigma, radius);
    cudaDeviceSynchronize();

    // Flip it back to normal orientation
    transpose(inputImage);
    cudaDeviceSynchronize();

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::printf(" Elapsed time: %f seconds\n", elapsedTime / 1000.0);

    // Output the saved data
    save_image(inputImage, outputPath);
    free_image(inputImage);
    free_image(outputImage);
    return 0;
}
