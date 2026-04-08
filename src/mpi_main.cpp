#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <mpi.h>
#include "stb_image.h"
#include "stb_image_write.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;
    int imgWidth, imgHeight, pixelComponent;
    unsigned char *extractedPixels = nullptr;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Manager validates input arguments and loads image
    if (world_rank == 0) {    
        if (argc != 2)  {
            std::cout << "Usage: -n <number-of-processes>./" << argv[0] << " <image-file-name>" << std::endl;
        }
        char *image_filename = argv[2];

        // Extract pixels with pixelComponent=4 (red, green, blue, alpha)
        extractedPixels = stbi_load(image_filename, &imgWidth, &imgHeight, &pixelComponent, 4);
        if (!extractedPixels) {
            std::cerr << "Failed to load image" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast image width and height from the manager to all other processes
    MPI_Bcast(&imgWidth,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imgHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::cout << "Hello from rank " << world_rank << " of " << world_size << std::endl;

    MPI_Finalize();
    return 0;
}
