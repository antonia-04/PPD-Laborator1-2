#include "../header/ConvolutionRows.h"
#include <cstring>
#include <algorithm>

ConvolutionRows::ConvolutionRows(int N, int M, int K, int P, int **matrix, int **filter,
                                 Barrier& barrier)
        : barrier(barrier) // init referinta la bariera
{
    this->N = N;
    this->M = M;
    this->K = K;
    this->P = P;
    this->matrix = matrix;
    this->convolution_matrix = filter;
    this->threads = new thread[P];
}

ConvolutionRows::~ConvolutionRows() {
    delete[] threads;
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

        // calculam indicii ghost
        int prevGhostIdx = (start_idx == 0) ? 0 : start_idx - 1; // clamping sus
        int nextGhostIdx = (end_idx == N) ? N - 1 : end_idx;     // clamping jos

        // start thread-ul
        threads[i] = thread(&ConvolutionRows::computeD, this, start_idx, end_idx, prevGhostIdx, nextGhostIdx);

        start_idx = end_idx;
    }

    for (int i = 0; i < P; i++) {
        threads[i].join();
    }
}

void ConvolutionRows::computeD(int start_idx, int end_idx, int prev_ghost_idx, int next_ghost_idx) {
    const int half = K / 2;
    const size_t row_size_bytes = M * sizeof(int);

    // alocam bufferele "ghost" locale
    int *prevGhostRow = new int[M];
    int *nextGhostRow = new int[M];

    // copiem liniile de granita inainte de bariera
    memcpy(prevGhostRow, matrix[prev_ghost_idx], row_size_bytes);
    memcpy(nextGhostRow, matrix[next_ghost_idx], row_size_bytes);

    // sincronizam inainte de calculul in-place
    barrier.wait();


    // alocam bufferele pentru sliding window
    int *prevRow = new int[M];
    int *currRow = new int[M];
    int *nextRow = new int[M];
    int *resultRow = new int[M];

    // init fereastra
    memcpy(prevRow, prevGhostRow, row_size_bytes);
    memcpy(currRow, matrix[start_idx], row_size_bytes);

    if (start_idx + 1 < end_idx) {
        // daca avem cel putin 2 linii in sectiune
        memcpy(nextRow, matrix[start_idx + 1], row_size_bytes);
    } else {
        // daca bucata are o singura linie, nextRow e ghost-ul de jos
        memcpy(nextRow, nextGhostRow, row_size_bytes);
    }

    // parcurgem liniile alocate acestei thread
    for (int i = start_idx; i < end_idx; i++) {

        for (int j = 0; j < M; j++) {
            int sum = 0;
            for (int a = 0; a < K; a++) {
                for (int b = 0; b < K; b++) {
                    int y = j + (b - half);
                    if (y < 0) y = 0;
                    if (y >= M) y = M - 1;

                    int value;
                    if (a == 0) value = prevRow[y];
                    else if (a == 1) value = currRow[y];
                    else value = nextRow[y];

                    sum += value * convolution_matrix[a][b];
                }
            }
            resultRow[j] = sum;
        }

        // suprascriem in-place
        memcpy(matrix[i], resultRow, row_size_bytes);

        // actualizam fereastra, rotim pointerii
        std::swap(prevRow, currRow);
        std::swap(currRow, nextRow);

        int next_i = i + 2;

        if (next_i < end_idx) {
            // daca urmatorul rand e tot in bucata noastra
            memcpy(nextRow, matrix[next_i], row_size_bytes);
        } else {
            // daca am ajuns la final, urmatorul rand e ghost-ul de jos
            memcpy(nextRow, nextGhostRow, row_size_bytes);
        }
    }

    // eliberam memoria alocata
    delete[] prevGhostRow;
    delete[] nextGhostRow;
    delete[] prevRow;
    delete[] currRow;
    delete[] nextRow;
    delete[] resultRow;
}