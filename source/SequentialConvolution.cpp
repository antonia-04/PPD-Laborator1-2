#include "../header/SequentialConvolution.h"
#include <cstring> // Pentru memcpy
#include <algorithm> // Pentru std::swap

SequentialConvolution::SequentialConvolution(int N, int M, int K, int **matrix, int **filter) {
    this->N = N;
    this->M = M;
    this->K = K;
    this->matrix = matrix;
    this->convolution_matrix = filter;
}


void SequentialConvolution::compute(const string &result_file) {
    const int half = K / 2;
    const size_t row_size_bytes = M * sizeof(int);

    // alocam bufferele O(M)
    int *prevRow = new int[M];
    int *currRow = new int[M];
    int *nextRow = new int[M];
    int *resultRow = new int[M];

    // init fereastra pt prima linie
    memcpy(prevRow, matrix[0], row_size_bytes);
    memcpy(currRow, matrix[0], row_size_bytes);
    if (N > 1) {
        memcpy(nextRow, matrix[1], row_size_bytes);
    } else {
        memcpy(nextRow, matrix[0], row_size_bytes);
    }

    for (int i = 0; i < N; i++) {

        // convolutia pentru linia i
        for (int j = 0; j < M; j++) {
            int sum = 0;
            for (int a = 0; a < K; a++) {
                for (int b = 0; b < K; b++) {

                    int y = j + (b - half); // y = j-1, j, j+1 (pt k=3)

                    // clamping stanga/dreapta
                    if (y < 0) y = 0;
                    if (y >= M) y = M - 1;

                    int value;
                    if (a == 0) value = prevRow[y];      // i-1
                    else if (a == 1) value = currRow[y]; //  i
                    else value = nextRow[y];             //  i+1

                    sum += value * convolution_matrix[a][b];
                }
            }
            resultRow[j] = sum;
        }

        // suprascriem in-place linia i
        memcpy(matrix[i], resultRow, row_size_bytes);

        // actualizam fereastra, rotim pointerii
        std::swap(prevRow, currRow);
        std::swap(currRow, nextRow);

        // copiem noua linie nextRow
        int next_i = (i + 2 < N) ? i + 2 : N - 1; // clamping jos
        memcpy(nextRow, matrix[next_i], row_size_bytes);
    }

    // dealoc bufferele
    delete[] prevRow;
    delete[] currRow;
    delete[] nextRow;
    delete[] resultRow;
}