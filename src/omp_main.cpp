/**
 * This version uses OpenMP for multi-core CPU parallelization.
 */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cstddef>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <vector>

#include "stb_image.h"
#include "stb_image_write.h"

// Basic image struct
struct Image {
	int width = 0;
	int height = 0;
	int channels = 0;
	std::byte *data = nullptr;
	bool transposed = false;
};

// Loads image into memory
Image create_image(const std::string &file) {
	Image image;
	image.data = reinterpret_cast<std::byte *>(stbi_load( file.c_str(), &image.width, &image.height, &image.channels, 4));

	if (!image.data) {
		std::cerr << "Failed to load image" << std::endl;
		std::exit(1);
		assert(image.data != nullptr && "Failed to load image");
	}

	image.channels = 4; // Force RGBA
	return image;
}

void free_image(Image &image) {
	if (image.data)
		stbi_image_free(image.data);
	image.data = nullptr;
	image.width = 0;
	image.height = 0;
	image.channels = 0;
	image.transposed = false;
}

// Transpose the image pixels to enable vertical blur with a horizontal kernel
void transpose(Image &image) {
	// Skip if no data
	if (!image.data)
		return;

	// Use STBI_MALLOC so stbi_image_free can handle it later
	std::byte *newData = reinterpret_cast<std::byte *>(STBI_MALLOC(image.width * image.height * image.channels));

    // Parallelize the transpose operation across CPU cores
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < image.width; i++) {
		for (int j = 0; j < image.height; j++) {
			for (int k = 0; k < image.channels; k++) {
				newData[(j + i * image.height) * image.channels + k] = image.data[(i + j * image.width) * image.channels + k];
			}
		}
	}

	stbi_image_free(image.data);
	image.data = newData;

	std::swap(image.width, image.height);
	image.transposed = !image.transposed;
}

void save_image(Image &image, const std::string &file) {
	if (!image.data)
		return;

	stbi_write_png(file.c_str(), image.width, image.height, image.channels, image.data, 0);
}

// Applies a 1D Gaussian blur horizontally
void horizontal_blur(Image &image, float sigma, int radius) {
	// Split the image into rows to calculate the row-based blur
	std::vector<std::byte> output_image(image.width * image.height * image.channels);

    // Parallelize row processing
	#pragma omp parallel for
	for (int y = 0; y < image.height; y++) {
		for (int x = 0; x < image.width; x++) {
			for (int k = 0; k < image.channels; k++) {
				// Calculate index of this pixel
				int index = (y * image.width + x) * image.channels + k;
				float sum = 0.0f;
				float weightSum = 0.0f;

				// 1D Gaussian kernel
				for (int dx = -radius; dx <= radius; dx++) {
					int nx = x + dx;

					if (nx >= 0 && nx < image.width) {
						float weight = std::exp(-(dx * dx) / (2.0f * sigma * sigma));
						int neighborIndex = (y * image.width + nx) * image.channels + k;
						sum += static_cast<float>(static_cast<unsigned char>(image.data[neighborIndex])) * weight;
						weightSum += weight;
					}
				}

				output_image[index] = static_cast<std::byte>(static_cast<unsigned char>(sum / weightSum));
			}
		}
	}

	memcpy(image.data, output_image.data(), output_image.size());
}

int main(int argc, char **argv) {
	Image image;

	std::string outputPath = "./resources/output.png";
	float sigma = 1.0f;
	int radius = 2;
	int num_threads = omp_get_num_procs();

	if (argc < 2) {
		std::printf("Usage: %s <image-file-name> [output-file-name] [sigma] [radius] [num-threads]\n", argv[0]);
		return 1;
	}
	if (argc > 2) outputPath = argv[2];
	if (argc > 3) sigma = std::stof(argv[3]);
	if (argc > 4) radius = std::stoi(argv[4]);
	if (argc > 5) num_threads = std::stoi(argv[5]);

	omp_set_num_threads(num_threads);

	std::printf("Running blur on %s with sigma %f and radius %d with OMP. Using %d threads.", argv[1], sigma, radius, num_threads);

	// Extract pixels with pixelComponent=4 (red, green, blue, alpha)
	image = create_image(argv[1]);

	double start_time = omp_get_wtime();

	// First pass: Horizontal blur
	horizontal_blur(image, sigma, radius);

	// Transpose to set up for vertical blur
	transpose(image);

	// Second pass: Horizontal blur (effectively vertical)
	horizontal_blur(image, sigma, radius);

	// Transpose back
	transpose(image);

	double elapsed_time = omp_get_wtime() - start_time;
	std::printf(" Elapsed time: %.4f seconds\n", elapsed_time);

	save_image(image, outputPath);
	free_image(image);
	return 0;
}
