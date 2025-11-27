#include <stdio.h>
#include <math.h>
#include <omp.h>

// 1. Разложение числа на сумму двух квадратов (первая пара)
int decomposition_sum_squares(int n, int *a, int *b){
    for (int i = 0; i * i <= n; i++){
        for (int j = 0; j * j <= n; j++){
            if (i * i + j * j == n) {
                *a = i;
                *b = j;
                return 1; // Найдена пара
            }
        }
    }
    return 0; // Пара не найдена
}

// 2. Нахождение n чисел Фибоначчи
void fibonacci(int n, int fib[]){
    fib[0] = 0;
    if (n > 1) fib[1] = 1;
    for (int i = 2; i < n; i++){
        fib[i] = fib[i-1] + fib[i-2];
    }
}

// 3. Нахождение n-го простого числа
int is_prime(int x){
    if (x < 2) return 0;
    for (int i = 2; i <= sqrt(x); i++){
        if (x % i == 0) return 0;
    }
    return 1;
}

int nth_prime(int n){
    int count = 0;
    int num = 2;
    while(1){
        if (is_prime(num)){
            count++;
            if (count == n) return num;
        }
        num++;
    }
}

// 4. Сумма всех делителей числа
int sum_divisors(int n){
    int sum = 0;
    for (int i = 1; i <= n; i++){
        if (n % i == 0) sum += i;
    }
    return sum;
}

int main(){
    int n = 25;

    int a = 0, b = 0;
    int fib_count = 10;
    int fib[10];

    int prime_index = 10;
    int prime_n;

    int sum_div;

#pragma omp parallel sections
    {
        #pragma omp section
        {
            if (decomposition_sum_squares(n, &a, &b))
                printf("Разложение %d на сумму квадратов: %d^2 + %d^2\n", n, a, b);
            else
                printf("Разложение не найдено\n");
        }
        #pragma omp section
        {
            fibonacci(fib_count, fib);
            printf("Первые %d чисел Фибоначчи: ", fib_count);
            for (int i = 0; i < fib_count; i++) printf("%d ", fib[i]);
            printf("\n");
        }
        #pragma omp section
        {
            prime_n = nth_prime(prime_index);
            printf("%d-е простое число: %d\n", prime_index, prime_n);
        }
        #pragma omp section
        {
            sum_div = sum_divisors(n);
            printf("Сумма делителей числа %d: %d\n", n, sum_div);
        }
    }
    return 0;
}