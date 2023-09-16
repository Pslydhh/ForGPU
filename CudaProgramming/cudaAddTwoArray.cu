#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

// There have strange access denied caused by hide this CHECK.
void CHECK(cudaError_t call) {
    const cudaError_t error_code = call;
    if (error_code != cudaSuccess) {
        printf("CUDA Error: \n");
        printf("File: %s\n", __FILE__);
        printf("Line: %d\n", __FILE__);
        printf("Error text: %s\n", cudaGetErrorString(error_code));
        exit(1);
    }
}

void checkLastError() {
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }
}

double __device__ add1_device(const double x, const double y) {
    return x + y;
}

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double* x, const double* y, double* z, const int);
void check(const double* z, const int N);

int main(void) {
    const int N = 10000001;
    const int M = sizeof(double) * N;
    double* x = (double*)malloc(M);
    double* y = (double*)malloc(M);
    double* z = (double*)malloc(M);

    for (int n = 0; n < N; ++n) {
        x[n] = a;
        y[n] = b;
    }

    double* d_x, * d_y, * d_z;
    CHECK(cudaMalloc((void**)&d_x, M));
    CHECK(cudaMalloc((void**)&d_y, M));
    CHECK(cudaMalloc((void**)&d_z, M));
    CHECK(cudaMemcpy(d_x, x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, y, M, cudaMemcpyHostToDevice));

    const int block_size = 128;
    const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    add << <grid_size, block_size >> > (d_x, d_y, d_z, N);
    checkLastError();

    CHECK(cudaMemcpy(z, d_z, M, cudaMemcpyDeviceToHost));

    check(z, N);

    free(x);
    free(y);
    free(z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

void __global__ add(const double* x, const double* y, double* z, const int N) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= N) {
        return;
    }
    z[n] = add1_device(x[n], y[n]);
}

void check(const double* z, const int N) {
    bool has_error = false;
    for (int n = 0; n < N; ++n) {
        if (fabs(z[n] - c) > EPSILON) {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}
