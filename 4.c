#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define MAX_NUM 200
#define MAX_SQUARES 100

typedef struct {
    int number;
    int squares[MAX_SQUARES][2];
    int squares_count;
    long long fibonacci[MAX_NUM];
    int fibonacci_count;
    int nth_prime;
    int sum_divisors;
} NumberResults;

NumberResults results[MAX_NUM];

// 1. Разложение числа на сумму ДВУХ квадратов
void decomposition_sum_squares(int n, int index) {
    results[index].squares_count = 0;
    for (int i = 0; i * i <= n; ++i) {
        int remaining = n - i * i;
        int j = (int)(sqrt(remaining) + 0.5);
        if (j * j == remaining && results[index].squares_count < MAX_SQUARES) {
            results[index].squares[results[index].squares_count][0] = i;
            results[index].squares[results[index].squares_count][1] = j;
            results[index].squares_count++;
        }
    }
}

// 2. Нахождение n чисел Фибоначчи
void calculate_n_fibonacci(int n, int index) {
    if (n <= 0) {
        results[index].fibonacci_count = 0;
        return;
    }
    results[index].fibonacci_count = n;
    results[index].fibonacci[0] = 0;
    if (n == 1) return;
    results[index].fibonacci[1] = 1;
    for (int i = 2; i < n; ++i) {
        results[index].fibonacci[i] = results[index].fibonacci[i-1] + results[index].fibonacci[i-2];
    }
}

// 3. Проверка простоты
int is_prime(int x) {
    if (x < 2) return 0;
    if (x == 2 || x == 3) return 1;
    if (x % 2 == 0 || x % 3 == 0) return 0;
    for (int i = 5; i * i <= x; i += 6) {
        if (x % i == 0 || x % (i + 2) == 0) return 0;
    }
    return 1;
}

// 4. N-ое простое число
int find_nth_prime(int n) {
    int count = 0, num = 2;
    while (count < n) {
        if (is_prime(num)) {
            count++;
            if (count == n) return num;
        }
        num++;
    }
    return -1;
}

// 5. Сумма делителей
int calculate_sum_divisors(int n) {
    int sum = 1 + (n != 1 ? n : 0);
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i) sum += n / i;
        }
    }
    return sum;
}

void process_number(int num, int index) {
    results[index].number = num;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            decomposition_sum_squares(num, index);
        }

        #pragma omp section
        {
            calculate_n_fibonacci(num, index);
        }

        #pragma omp section
        {
            results[index].nth_prime = find_nth_prime(num);
        }

        #pragma omp section
        {
            results[index].sum_divisors = calculate_sum_divisors(num);
        }
    }
}

void print_results(int n) {
    for (int i = 0; i < n; i++) {
        printf("=== ЧИСЛО %d ===\n", results[i].number);

        printf("Разложение на сумму квадратов:\n");
        for (int j = 0; j < results[i].squares_count; j++) {
            printf("%d = %d² + %d²\n",
                   results[i].number,
                   results[i].squares[j][0],
                   results[i].squares[j][1]);
        }

        printf("Первые %d чисел Фибоначчи: ", results[i].fibonacci_count);
        for (int j = 0; j < results[i].fibonacci_count; j++) {
            printf("%lld ", results[i].fibonacci[j]);
        }
        printf("\n");

        printf("%d-е простое число: %d\n", results[i].number, results[i].nth_prime);
        printf("Сумма делителей: %d\n", results[i].sum_divisors);
        printf("\n");
    }
}

int main() {
    int n;
    printf("Размер массива: ");
    scanf("%d", &n);

    srand(time(NULL));
    int array[n];

    printf("Массив: ");
    for (int i = 0; i < n; i++) {
        array[i] = rand() % MAX_NUM + 1;
        printf("%d ", array[i]);
    }
    printf("\n\n");

    for (int i = 0; i < n; i++) {
        process_number(array[i], i);
    }

    print_results(n);

    return 0;
}