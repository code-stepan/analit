#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

double calculate_pi(long long num_points) {
    long long points_inside = 0;
    double x, y;

    #pragma omp parallel private(x, y) reduction(+:points_inside)
    {
        unsigned int seed = (unsigned int)time(NULL) ^ omp_get_thread_num();
        
        #pragma omp for
        for (long long i = 0; i < num_points; i++) {
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;
            
            if (x * x + y * y <= 1.0) {
                points_inside++;
            }
        }
    }

    return 4.0 * (double)points_inside / (double)num_points;
}

int main() {
    double epsilon;
    printf("Введите погрешность (например, 0.001): ");
    
    if (scanf("%lf", &epsilon) != 1 || epsilon <= 0) {
        printf("Ошибка: введите положительное число для погрешности.\n");
        return 1;
    }

    if (epsilon < 1e-10) {
        printf("Предупреждение: установлена очень высокая точность. Вычисления могут занять много времени.\n");
    }

    long long num_points = 1000;
    double pi_old = 0.0;
    double pi_new = 0.0;
    int iterations = 0;
    const double real_pi = M_PI;
    const long long max_points = 1000000000LL;
    const int max_iterations = 50;

    clock_t start_time = clock();

    do {
        pi_old = pi_new;
        pi_new = calculate_pi(num_points);
        
        double diff_from_real = fabs(pi_new - real_pi);
        double diff_from_previous = fabs(pi_new - pi_old);
        
        iterations++;
        
        printf("Итерация %d:\n", iterations);
        printf("  Количество точек: %lld\n", num_points);
        printf("  Вычисленное π: %.15f\n", pi_new);
        printf("  Погрешность от реального: %.10f\n", diff_from_real);
        printf("  Изменение от предыдущего: %.10f\n", diff_from_previous);
        
        // Проверка на достижение предела
        if (num_points > max_points / 2) {
            printf("\nПредупреждение: достигнуто максимальное количество точек.\n");
            break;
        }
        
        if (iterations >= max_iterations) {
            printf("\nПредупреждение: достигнуто максимальное количество итераций.\n");
            break;
        }
        
        num_points *= 2;
        printf("\n");
        
    } while (fabs(pi_new - pi_old) > epsilon);

    clock_t end_time = clock();
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    double final_diff = fabs(pi_new - real_pi);
    
    printf("\n=== РЕЗУЛЬТАТЫ ===\n");
    printf("Финальное значение π: %.15f\n", pi_new);
    printf("Реальное значение π:  %.15f\n", real_pi);
    printf("Абсолютная погрешность: %.10f\n", final_diff);
    printf("Относительная погрешность: %.6f%%\n", (final_diff / real_pi) * 100);
    printf("Количество итераций: %d\n", iterations);
    printf("Финальное количество точек: %lld\n", num_points / 2);
    printf("Время выполнения: %.3f секунд\n", execution_time);
    
    // Оценка точности
    if (final_diff < 1e-6) {
        printf("Точность: очень высокая (< 1e-6)\n");
    } else if (final_diff < 1e-4) {
        printf("Точность: высокая (< 1e-4)\n");
    } else if (final_diff < 1e-2) {
        printf("Точность: средняя (< 1e-2)\n");
    } else {
        printf("Точность: низкая\n");
    }

    return 0;
}