#include <stdio.h>

__global__ void hello_kernel() {
    const int thid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Hello from thread %d!\n", thid);
}

int main()
{
    cudaSetDevice(0);
    hello_kernel << <2, 4 >> > ();
    cudaDeviceSynchronize();
}
