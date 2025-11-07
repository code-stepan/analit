#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

void printMatrix(float *matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f\t", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Ядро CUDA
__global__ void matrixMultiply(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    srand(time(0));
    int N = 4; // Размер матрицы N x N
    size_t size = N * N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1 + rand() % 9; // Значения от 1 до 9
        h_B[i] = 1 + rand() % 9;
    }

    printf("Матрица A:\n");
    printMatrix(h_A, N);
    printf("Матрица B:\n");
    printMatrix(h_B, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Результирующая матрица C:\n");
    printMatrix(h_C, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}