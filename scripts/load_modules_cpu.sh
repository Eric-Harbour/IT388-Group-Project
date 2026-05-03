#!/bin/bash
# Load the necessary modules for compiling and running CPU-based (OMP/MPI) code
echo "Loading modules..."
echo "GCC"
module load gcc/10.2.0

echo "CMake"
module load cmake/3.21.4

echo "OpenMPI"
module load openmpi/4.1.3