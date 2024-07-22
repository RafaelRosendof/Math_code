#include <iostream>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

// Função CUDA para cálculo
__device__ double f(double x) {
    return log10(sqrt(x*x + x) + 1);
}

// Kernel CUDA
__global__ void trapezoidal_rule_cuda(double a, double b, int n, double h, double* integral) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double temp_integral = 0.0;

    if (idx == 0) {
        temp_integral += f(a);
    }
    if (idx == n) {
        temp_integral += f(b);
    }
    if (idx > 0 && idx < n) {
        temp_integral += 2.0 * f(a + idx * h);
    }

    atomicAdd(integral, temp_integral);
}

// Função host para chamar o kernel
double trapezoidal_rule_cuda_host(double a, double b, int n) {
    double h = (b - a) / n;
    double* d_integral;
    double integral = 0.0;
    
    cudaMalloc((void**)&d_integral, sizeof(double));
    cudaMemcpy(d_integral, &integral, sizeof(double), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    trapezoidal_rule_cuda<<<numBlocks, blockSize>>>(a, b, n, h, d_integral);
    cudaMemcpy(&integral, d_integral, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_integral);

    integral *= h / 2.0;
    return integral;
}

int main() {
    double a = 2;
    double b = 100;
    int n = 1000000; // Aumentar o número de subintervalos para maior precisão

    double integral = trapezoidal_rule_cuda_host(a, b, n);
    
    std::cout << "Integral (CUDA Trapézio): " << std::setprecision(30) << integral << std::endl;

    return 0;
}
