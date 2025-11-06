#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double calculate_pi(long long n) {
    double h = 1.0 / n;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        double x = (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * h;
}

int main() {
    double epsilon;
    printf("Введие epsilon: ");
    scanf("%lf", &epsilon);
    
    if (epsilon <= 0) {
        printf("Погрещность не может быть меньше нуля!");
        return 1;
    }

    long long n = 1000; // интервалы
    double pi_prev = 0.0;
    double pi_current = calculate_pi(n);
    int iteration = 1;

    printf("Итерация %2d: n = %10lld, π = %.10f\n", iteration, n, pi_current);

    while (fabs(pi_current - pi_prev) >= epsilon) {
        pi_prev = pi_current;
        n *= 2;
        pi_current = calculate_pi(n);
        iteration++;

        printf("Итерация %2d: n = %10lld, π = %.20f, разность = %.20f\n", 
               iteration, n, pi_current, fabs(pi_current - pi_prev));
        
        if (iteration > 50) {
            printf("Достигнуто максимальное количество итераций!\n");
            break;
        }
    }

    printf("\nРезультат:\n");
    printf("π = %.20f с точностью %.20f\n", pi_current, epsilon);
    printf("Количество интервалов: %lld\n", n);
    printf("Количество итераций: %d\n", iteration);

    return 0;
}