#include <math.h>
#include <stdio.h>
#include "cuda_runtime.h"

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void vecAddKernel(double* A, double* B, double* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void check(const double* z, const int N) {
    bool has_error = false;
    for (int i = 0; i < N; ++i) {
        if (fabs(z[i] - c) > EPSILON) {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

void vecAdd(double* A, double* B, double* C, int n) {
    double* A_d, * B_d, * C_d;
    int size = n * sizeof(double);

    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    vecAddKernel << <ceil(n / 1024.0), 1024 >> > (A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    const int N = 10000000;
    double* h_x = (double*)malloc(N * sizeof(double));
    double* h_y = (double*)malloc(N * sizeof(double));
    double* h_z = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        h_x[i] = a;
        h_y[i] = b;
    }

    vecAdd(h_x, h_y, h_z, N);

    check(h_z, N);
}
