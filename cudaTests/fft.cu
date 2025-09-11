#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

//#define N 1024            // número de pontos espaciais
//#define T_FINAL 10.0      // tempo final em segundos
//#define DT 0.0001         // passo de tempo
//#define D 0.1f            // coeficiente de difusão

__global__ void apply_diffusion(cufftComplex* u_hat, int N, float DT, float D) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    // índice de frequência de Fourier: [-N/2, ..., N/2]
    int shifted_k = (k <= N/2) ? k : k - N;
    float k2 = shifted_k * shifted_k;
    float factor = expf(-D * k2 * DT);
    u_hat[k].x *= factor;
    u_hat[k].y *= factor;
}

__global__ void normalize_ifft(cufftComplex* u, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        u[i].x /= N;
        u[i].y /= N;
    }
}

int main() {
    // Aloca memória no host
    const int N = 1024;            // número de pontos espaciais
    const float T_FINAL = 10.0f;   // tempo final em segundos
    const float DT = 0.0001f;      // passo de tempo
    const float D = 0.1f;          // coeficiente de difusão

    cufftComplex *h_u = new cufftComplex[N];
    float dx = 1.0f / N;

    // Inicializa condição inicial: sin(2πx)
    for (int i = 0; i < N; ++i) {
        float x = i * dx;
        h_u[i].x = sinf(2 * M_PI * x); // parte real
        h_u[i].y = 0.0f;               // parte imaginária
    }

    // Aloca no device
    cufftComplex *d_u;
    cudaMalloc(&d_u, sizeof(cufftComplex) * N);
    cudaMemcpy(d_u, h_u, sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);

    // Cria plano FFT
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    // Transforma para o domínio de Fourier
    cufftExecC2C(plan, d_u, d_u, CUFFT_FORWARD);

    // Loop de tempo
    int steps = T_FINAL / DT;
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    for (int t = 0; t < steps; ++t) {
        apply_diffusion<<<gridSize, blockSize>>>(d_u, N, DT, D);
    }

    // Inversa para obter u(x,T)
    cufftExecC2C(plan, d_u, d_u, CUFFT_INVERSE);
    normalize_ifft<<<gridSize, blockSize>>>(d_u, N);

    // Copia de volta para o host
    cudaMemcpy(h_u, d_u, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);

    // Exibe resultados
    for (int i = 0; i < 10; ++i) {
        printf("u[%d] = %.5f\n", i, h_u[i].x);
    }

    // Libera recursos
    cufftDestroy(plan);
    cudaFree(d_u);
    delete[] h_u;

    return 0;
}
