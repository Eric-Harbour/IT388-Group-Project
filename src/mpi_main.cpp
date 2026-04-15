#include <cstdio>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <cstddef>
#include <cassert>
#include <mpi.h>
#include "stb_image.h"
#include "stb_image_write.h"

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::byte* data = nullptr;
    bool transposed = false;
};

Image create_image(const std::string& file){
    Image image;
    image.data = reinterpret_cast<std::byte*>(stbi_load(file.c_str(), &image.width, &image.height, &image.channels, 4));

    if (!image.data) {
        std::cerr << "Failed to load image" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        assert(image.data != nullptr && "Failed to load image");
    }

    image.channels = 4; // Reset channels to 4 because stbi_load's req_comp is 4 so the data will be forced to 4.
    return image;
}

void free_image(Image& image){
    if(image.data) stbi_image_free(image.data);
    image.data = nullptr;
    image.width = 0;
    image.height = 0;
    image.channels = 0;
    image.transposed = false;
}

void transpose(Image& image) {
    // Skip if no data
    if(!image.data)
        return;

    // Use STBI_MALLOC so stbi_image_free can handle it later
    std::byte* newData = reinterpret_cast<std::byte*>(STBI_MALLOC(image.width * image.height * image.channels));

    for(int i = 0; i < image.width; i++) {
        for(int j = 0; j < image.height; j++) {
            for(int k = 0; k < image.channels; k++) {
                newData[(j + i * image.height) * image.channels + k] = image.data[(i + j * image.width) * image.channels + k];
            }
        }
    }

    stbi_image_free(image.data);
    image.data = newData;
    
    // std::swap(image.width, image.height);
    image.transposed = !image.transposed;
}

void save_image(Image& image, const std::string& file){
    if(!image.data)
        return;

    stbi_write_png(file.c_str(), image.width, image.height, image.channels, image.data, 0);
}

void horizontal_blur(Image& image, float sigma, int radius, int worldRank, int worldSize, MPI_Comm world){
    // Split the image into rows to calculate the row-based blur
    std::vector<int> sendCounts, displacements;
    sendCounts.resize(worldSize);
    displacements.resize(worldSize);
    
    for (int rank = 0; rank < worldSize; rank++) {
        int localHeight = image.height / worldSize;

        if(rank == worldSize - 1)
            localHeight += image.height % worldSize;

        sendCounts[rank] = localHeight * image.width * image.channels;
        displacements[rank] = (rank == 0) ? 0 : displacements[rank - 1] + sendCounts[rank - 1];
    }

	int localSize = sendCounts[worldRank]; // How many bytes each processor handles
	int localHeight = localSize / (image.width * image.channels);

    // Scatter pixels so each processor has their own subdivided image
    std::byte* localPixels = new std::byte[localSize];
    MPI_Scatterv(image.data, sendCounts.data(), displacements.data(), MPI_UNSIGNED_CHAR, localPixels, localSize, MPI_UNSIGNED_CHAR, 0, world);

    // Blur the local pixels horizontally
    for(int y = 0; y < localHeight; y++){
        for(int x = 0; x < image.width; x++){
            for(int k = 0; k < image.channels; k++){
                // Calculate index of this pixel
                int index = (y * image.width + x) * image.channels + k;
                float sum = 0.0f;

                // Gaussian blur kernel
                float weightSum = 0.0f;

                // Blur this row by radius
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;

                    if (nx >= 0 && nx < image.width) {
                        float weight = std::exp(-(dx * dx) / (2.0f * sigma * sigma));
                        int neighborIndex = (y * image.width + nx) * image.channels + k;
                        sum += static_cast<float>(static_cast<unsigned char>(localPixels[neighborIndex])) * weight;
                        weightSum += weight;
                    }
                }

                localPixels[index] = static_cast<std::byte>(static_cast<unsigned char>(sum / weightSum));
            }
        }
    }

    // Send the local pixels back to the image
    MPI_Gatherv(localPixels, localSize, MPI_UNSIGNED_CHAR, image.data, sendCounts.data(), displacements.data(), MPI_UNSIGNED_CHAR, 0, world);
    delete[] localPixels;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int worldRank, worldSize;
    Image image;
    
    std::string outputPath = "./resources/output.png";
    float sigma = 1.0f;
    int radius = 2;

    MPI_Comm world = MPI_COMM_WORLD;
    MPI_Comm_rank(world, &worldRank);
    MPI_Comm_size(world, &worldSize);

    // Manager validates input arguments and loads image
    if (worldRank == 0) {    
        if (argc < 2) {
            std::printf("Usage: mpiexec -n <number-of-processes> %s (image-file-name) [output-file-name] [sigma] [radius]\n", argv[0]);
            MPI_Abort(world, 1);
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

        std::printf("Running blur on %s with sigma %f and radius %d\n", argv[1], sigma, radius);

        // Extract pixels with pixelComponent=4 (red, green, blue, alpha)
        image = create_image(argv[1]);
    }

    // Time the execution
    double startTime = MPI_Wtime();

    // Broadcast basic image data
    MPI_Bcast(&image.width, 1, MPI_INT, 0, world);
    MPI_Bcast(&image.height, 1, MPI_INT, 0, world);
    MPI_Bcast(&image.channels, 1, MPI_INT, 0, world);
    
    // Do the row gaussian blur first
    horizontal_blur(image, sigma, radius, worldRank, worldSize, world);

    // Flip the image and do the vertical gaussian bllur now
    transpose(image);

    // Now that its flipped, do another blur for the other direction
    horizontal_blur(image, sigma, radius, worldRank, worldSize, world);

    // Reconstruct back to normal image
    transpose(image);

    // Feedback for time
    double endTime = MPI_Wtime();
    double elapsedTime = endTime - startTime;
    
    if(worldRank == 0){
        std::printf("Time taken: %f seconds\n", elapsedTime);
    }

    // Cleanup and save the output
    save_image(image, outputPath);
    MPI_Finalize();
    return 0;
}
