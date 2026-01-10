#include "../header/ConvolutionRows.h"
#include "../header/DataGeneration.h"

ConvolutionRows::ConvolutionRows(int N, int M, int K, int P, int **matrix, int **filter) {
    this->N = N;
    this->M = M;
    this->K = K;
    this->P = P;
    this->matrix = matrix;
    this->convolution_matrix = filter;
    this->threads = new thread[P];
    this->new_matrix = new int *[N];
    for (int i = 0; i < N; i++) {
        this->new_matrix[i] = new int[M];
    }
}

void ConvolutionRows::run() {


    int rows_per_thread = N / P;
    int extra = N % P;
    int start_idx = 0;


    for (int i = 0; i < P; i++) {
        int end_idx = start_idx + rows_per_thread;
        if (extra > 0) {
            extra--;
            end_idx++;
        }


        threads[i] = thread([this, start_idx, end_idx]() {
            this->computeD(start_idx, end_idx);
        });

        start_idx = end_idx;
    }


    for (int i = 0; i < P; i++) {
        threads[i].join();
    }
}

void ConvolutionRows::computeD(int start_idx, int end_idx) {
    for (int i = start_idx; i < end_idx; i++) {
        for (int j = 0; j < M; j++) {
            new_matrix[i][j] = compute_element(i, j);
        }
    }
}

int ConvolutionRows::compute_element(int i, int j) {
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

int **ConvolutionRows::getNewMatrix() {
    return new_matrix;
}