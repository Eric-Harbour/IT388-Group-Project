#include <iostream>

__global__ void hello_kernel() {
    printf("Hello from CUDA thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    hello_kernel<<<1, 5>>>();
    cudaDeviceSynchronize();

    std::cout << "CUDA execution finished." << std::endl;
    return 0;
}
