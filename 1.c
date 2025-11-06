#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 1000

double** allocate_matrix(int size) {
    double **matrix = (double**)malloc(size * sizeof(double*));
    double *data = (double*)malloc(size * size * sizeof(double));
    
    if (matrix == NULL || data == NULL) {
        fprintf(stderr, "Ошибка выделения памяти!\n");
        free(matrix);
        free(data);
        return NULL;
    }
    
    for (int i = 0; i < size; i++) {
        matrix[i] = data + i * size;
    }
    
    return matrix;
}

void free_matrix(double** matrix) {
    if (matrix != NULL) {
        free(matrix[0]);
        free(matrix);
    }
}

void multiply_matrices(int threads_count) {
    double start_time, end_time;

    double **A = allocate_matrix(N);
    double **B = allocate_matrix(N);
    double **C = allocate_matrix(N);
    
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Неудалось выделить память для матриц!\n");
        free_matrix(A);
        free_matrix(B);
        free_matrix(C);
        return;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }
    }

    start_time = omp_get_wtime();

    #pragma omp parallel for num_threads(threads_count)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    end_time = omp_get_wtime();

    printf("Threads: %d, Time: %.6f seconds\n", threads_count, end_time - start_time);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
}

int main() {
    printf("=== Тестирование умножения матриц %dx%d ===\n", N, N);
    printf("Используемая память: ~%.1f МБ\n", (3 * N * N * sizeof(double)) / (1024.0 * 1024.0));
    printf("========================================\n");
    
    srand(time(NULL));

    int threads[] = {1, 2, 4, 8, 16};

    for (int i = 0; i < 5; i++) {
        multiply_matrices(threads[i]);
    }

    printf("========================================\n");
    printf("Тестирование завершено\n");
    
    return 0;
}
