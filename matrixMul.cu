#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 5000
#define BLOCK_SIZE 32

__global__ void matrixMultiply(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void fillRandomMatrix(float* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

int main() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error before start: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // информация
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "=== CUDA Device ===" << std::endl;
    std::cout << "Name: " << prop.name << std::endl;
    std::cout << "Compute: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;

    int matrixSize = N * N;
    size_t bytes = matrixSize * sizeof(float);

    // зост память
    float *h_A = new float[matrixSize];
    float *h_B = new float[matrixSize];
    float *h_C = new float[matrixSize];

    std::cout << "\nMatrix: " << N << "x" << N << " (" << bytes/(1024*1024) << " MB)" << std::endl;

    // заполнение случайными числами
    fillRandomMatrix(h_A, matrixSize);
    fillRandomMatrix(h_B, matrixSize);
    
    std::cout << "A[0][0] = " << h_A[0] << ", B[0][0] = " << h_B[0] << std::endl;

    // устройство память
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // копируем на GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // конфигурация запуска
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "Launching kernel: " << blocksPerGrid.x << "x" << blocksPerGrid.y 
              << " blocks, " << threadsPerBlock.x << "x" << threadsPerBlock.y << " threads" << std::endl;

    // запуск ядра
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    double timeMs = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Time: " << timeMs << " ms" << std::endl;
    std::cout << "GFLOPS: " << (2.0 * N * N * N / 1e9) / (timeMs / 1000.0) << std::endl;

    // купируем обратно
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // вывод некторых элементов
    std::cout << "\nResult matrix C:" << std::endl;
    std::cout << "C[0][0] = " << h_C[0] << std::endl;
    std::cout << "C[0][1] = " << h_C[1] << std::endl;
    std::cout << "C[1][0] = " << h_C[N] << std::endl;
    std::cout << "C[1][1] = " << h_C[N+1] << std::endl;
    std::cout << "C[4999][4999] = " << h_C[(N-1)*N + (N-1)] << std::endl;


    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    std::cout << "\nDone!" << std::endl;
    return 0;
}