#include "../header/SequentialConvolution.h"

#include <pthread.h>

#include "../header/DataGeneration.h"

SequentialConvolution::SequentialConvolution(int N, int M, int K, int **matrix, int **filter) {
    this->N = N;
    this->M = M;
    this->K = K;
    this->matrix = matrix;
    this->convolution_matrix = filter;
    this->new_matrix = new int *[N];
    for (int i = 0; i < N; i++) {
        this->new_matrix[i] = new int[M];
    }
}


void SequentialConvolution::compute(const string &result_file) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            new_matrix[i][j] = compute_element(i, j);
        }
    }
}

int SequentialConvolution::compute_element(int i, int j) {
    const int half = K / 2;
    int sum = 0;

    for (int a = -half; a <= half; a++) {
        for (int b = -half; b <= half; b++) {
            int x = i + a;
            int y = j + b;
            if (x < 0) x = 0;
            if (y < 0) y = 0;
            if (x >= N) x = N - 1;
            if (y >= M) y = M - 1;
            sum += matrix[x][y] * convolution_matrix[a + half][b + half];
        }
    }
    return sum;
}

int **SequentialConvolution::getNewMatrix() {
    return new_matrix;
}

SequentialConvolution::~SequentialConvolution() {
    for (int i = 0; i < N; i++)
        delete[] new_matrix[i];
    delete[] new_matrix;
}
