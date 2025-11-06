#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include <math.h>

#define N 5000
#define BLOCK_SIZE 16

const char *matrixMultiplyKernelSource = "\n" \
"__kernel void matrixMultiply(__global const float *A, \n" \
"                             __global const float *B, \n" \
"                             __global float *C, \n" \
"                             const int n) { \n" \
"    int row = get_global_id(1); \n" \
"    int col = get_global_id(0); \n" \
"    if (row < n && col < n) { \n" \
"        float sum = 0.0f; \n" \
"        for (int k = 0; k < n; ++k) { \n" \
"            sum += A[row * n + k] * B[k * n + col]; \n" \
"        } \n" \
"        C[row * n + col] = sum; \n" \
"    } \n" \
"} \n";

int main() {
    // 1. Инициализация OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(1, &platform, &numPlatforms);
    if (err != CL_SUCCESS) {
        printf("Error getting platform IDs\n");
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting device IDs\n");
        return 1;
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating context\n");
        return 1;
    }

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating command queue\n");
        return 1;
    }

    // 2. Инициализация матриц
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_B = (float*)malloc(N * N * sizeof(float));
    float *h_C = (float*)malloc(N * N * sizeof(float));
    float *h_C_cpu = (float*)malloc(N * N * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / (float)(RAND_MAX) * 1.0f;
        h_B[i] = (float)rand() / (float)(RAND_MAX) * 1.0f;
    }

    // 3. Создание буферов на устройстве
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float), NULL, &err);

    // 4. Копирование данных на устройство
    err = clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, N * N * sizeof(float), h_A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, N * N * sizeof(float), h_B, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error writing buffers\n");
        return 1;
    }

    // 5. Создание и компиляция ядра
    cl_program program = clCreateProgramWithSource(context, 1, &matrixMultiplyKernelSource, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating program\n");
        return 1;
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("OpenCL build error:\n%s\n", log);
        free(log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "matrixMultiply", &err);
    if (err != CL_SUCCESS) {
        printf("Error creating kernel\n");
        return 1;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &N);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arguments\n");
        return 1;
    }

    // 6. Запуск ядра
    size_t global[2] = {N, N};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error enqueuing kernel\n");
        return 1;
    }

    // 7. Копирование результата обратно на хост
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, N * N * sizeof(float), h_C, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error reading buffer\n");
        return 1;
    }

    // 8. Проверка результата (сравнение с CPU)
    clock_t start_cpu = clock();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C_cpu[i * N + j] = sum;
        }
    }
    clock_t end_cpu = clock();
    double elapsed_cpu = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("CPU time: %f seconds\n", elapsed_cpu);

    // Сравнение результатов GPU и CPU
    int correct = 1;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_C[i] - h_C_cpu[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }
    printf("Matrix multiplication %s!\n", correct ? "succeeded" : "failed");

    // Очистка
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
