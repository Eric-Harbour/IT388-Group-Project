#!/bin/bash
# Load the necessary modules for CUDA development. 
# Note: These are kept separate from CPU modules to avoid environment conflicts on Expanse.
echo "Loading modules..."
echo "GPU"
module load gpu/0.17.3b

echo "CUDA"
module load cuda11.7/toolkit/11.7.1
