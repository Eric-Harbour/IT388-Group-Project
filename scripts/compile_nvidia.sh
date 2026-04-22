#!/bin/bash
nvcc -std=c++17 -o cuda_exec -O3 cuda_main.cu -I./../vendor/stb