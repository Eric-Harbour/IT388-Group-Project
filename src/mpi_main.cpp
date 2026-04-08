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

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::cout << "Hello from rank " << world_rank << " of " << world_size << std::endl;

    MPI_Finalize();
    return 0;
}
