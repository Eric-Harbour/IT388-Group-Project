#!/bin/bash
echo "Loading modules..."
echo "GCC"
module load gcc/10.2.0

echo "CMake"
module load cmake/3.21.4

echo "OpenMPI"
module load openmpi/4.1.3

echo "GPU"
module load gpu/0.15.4

echo "CUDA"
module load cuda11.7/toolkit/11.7.1
