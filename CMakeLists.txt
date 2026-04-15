cmake_minimum_required(VERSION 3.20)

project(IT388Project LANGUAGES CXX CUDA)

# Variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add user libraries
add_subdirectory(vendor/stb)

# OpenMP executable
find_package(OpenMP REQUIRED)

add_executable(omp_exec
    src/omp_main.cpp
)

target_link_libraries(omp_exec
    PRIVATE OpenMP::OpenMP_CXX stb
)

# MPI executable
find_package(MPI REQUIRED)

add_executable(mpi_exec
    src/mpi_main.cpp
)

target_link_libraries(mpi_exec
    PRIVATE MPI::MPI_CXX stb
)

# CUDA executable
enable_language(CUDA)

add_executable(cuda_exec
    src/cuda_main.cu
)

target_link_libraries(cuda_exec
	PRIVATE stb
)

set_target_properties(cuda_exec PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
)

target_compile_options(cuda_exec PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=550>
)