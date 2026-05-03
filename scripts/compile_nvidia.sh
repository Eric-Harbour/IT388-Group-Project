#!/bin/bash
# We use this manual nvcc command because CMake and the CUDA toolkit 
# couldn't be loaded at the same time on the Expanse cluster.
nvcc -std=c++17 -o cuda_exec -O3 cuda_main.cu -I./../vendor/stb