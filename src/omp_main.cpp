/**
 * Running omp_main.cpp requires the program to know.
 * 1. How many threads you want to run (argv[1]: nThreads)
 * 2. What image you want to process (argv[2]: imgName)
 */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cstddef>
#include <iostream>
#include <omp.h>
#include <vector>

#include "stb_image.h"
#include "stb_image_write.h"

struct Image {
	int width = 0;
	int height = 0;
	int channels = 0;
	std::byte *data = nullptr;
	bool transposed = false;
};

Image create_image(const std::string &file) {
	Image image;
	image.data = reinterpret_cast<std::byte *>(stbi_load(
		file.c_str(), &image.width, &image.height, &image.channels, 4));

	if (!image.data) {
		std::cerr << "Failed to load image" << std::endl;
		// TODO: exit somehow
		assert(image.data != nullptr && "Failed to load image");
	}

	image.channels = 4; // Reset channels to 4 because stbi_load's req_comp is 4
						// so the data will be forced to 4.
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

void transpose(Image &image) {
	// Skip if no data
	if (!image.data)
		return;

	// Use STBI_MALLOC so stbi_image_free can handle it later
	std::byte *newData = reinterpret_cast<std::byte *>(
		STBI_MALLOC(image.width * image.height * image.channels));

	for (int i = 0; i < image.width; i++) {
		for (int j = 0; j < image.height; j++) {
			for (int k = 0; k < image.channels; k++) {
				newData[(j + i * image.height) * image.channels + k] =
					image.data[(i + j * image.width) * image.channels + k];
			}
		}
	}

	stbi_image_free(image.data);
	image.data = newData;

	// std::swap(image.width, image.height);
	image.transposed = !image.transposed;
}

void save_image(Image &image, const std::string &file) {
	if (!image.data)
		return;

	stbi_write_png(file.c_str(), image.width, image.height, image.channels,
				   image.data, 0);
}

void horizontal_blur(Image &image, float sigma, int radius, int worldRank,
					 int worldSize) {
	// Split the image into rows to calculate the row-based blur
	std::vector<int> sendCounts, displacements;
	sendCounts.resize(worldSize);
	displacements.resize(worldSize);

	for (int rank = 0; rank < worldSize; rank++) {
		int localHeight = image.height / worldSize;

		if (rank == worldSize - 1)
			localHeight += image.height % worldSize;

		sendCounts[rank] = localHeight * image.width * image.channels;
		displacements[rank] =
			(rank == 0) ? 0 : displacements[rank - 1] + sendCounts[rank - 1];
	}

	int localSize =
		sendCounts[worldRank]; // How many bytes each processor handles
	int localHeight = localSize / (image.width * image.channels);

	// Scatter pixels so each processor has their own subdivided image
	std::byte *localPixels = new std::byte[localSize];

	// Blur the local pixels horizontally
#pragma omp parallel for
	for (int y = 0; y < localHeight; y++) {
		for (int x = 0; x < image.width; x++) {
			for (int k = 0; k < image.channels; k++) {
				// Calculate index of this pixel
				int index = (y * image.width + x) * image.channels + k;
				float sum = 0.0f;

				// Gaussian blur kernel
				float weightSum = 0.0f;

				// Blur this row by radius
				for (int dx = -radius; dx <= radius; dx++) {
					int nx = x + dx;

					if (nx >= 0 && nx < image.width) {
						float weight =
							std::exp(-(dx * dx) / (2.0f * sigma * sigma));
						int neighborIndex =
							(y * image.width + nx) * image.channels + k;
						sum += static_cast<float>(static_cast<unsigned char>(
								   localPixels[neighborIndex])) *
							   weight;
						weightSum += weight;
					}
				}

				localPixels[index] = static_cast<std::byte>(
					static_cast<unsigned char>(sum / weightSum));
			}
		}
	}

	delete[] localPixels;
}

int main(int argc, char **argv) {
	Image image;

	std::string outputPath = "./resources/output.png";
	float sigma = 1.0f;
	int radius = 2;
	int numThreads = omp_get_num_procs();

	if (argc < 2) {
		std::printf("Usage: %s <image-file-name> [output-file-name] [sigma] "
					"[radius] [num-threads]\n",
					argv[0]);
		return 1;
	}
	if (argc > 2) {
		outputPath = argv[2];
	}
	if (argc > 3) {
		sigma = std::stof(argv[3]);
	}
	if (argc > 4) {
		radius = std::stoi(argv[4]);
	}
	if (argc > 5) {
		int numThreads = std::stoi(argv[5]);
	}
	omp_set_num_threads(numThreads);

	std::printf("Running blur on %s with sigma %f and radius %d\n", argv[1],
				sigma, radius);

	// Extract pixels with pixelComponent=4 (red, green, blue, alpha)
	image = create_image(argv[1]);
	// Do the row gaussian blur first
	horizontal_blur(image, sigma, radius, omp_get_thread_num(), numThreads);

	// Flip the image and do the vertical gaussian bllur now
	transpose(image);

	// Now that its flipped, do another blur for the other direction
	horizontal_blur(image, sigma, radius, omp_get_thread_num(), numThreads);

	// Reconstruct back to normal image
	transpose(image);

	return 0;
}
