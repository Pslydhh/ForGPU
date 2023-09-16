#include <stdio.h>

__global__ void hello_kernel() {
    const int thid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Hello from thread %d\n", thid);
}

int main(int argc, char* argv[]) {
    cudaSetDevice(0);
    // invoke kernel using 4 threads executed in 1 thread block
    hello_kernel << <1, 4 >> > ();

    cudaDeviceSynchronize();
}
