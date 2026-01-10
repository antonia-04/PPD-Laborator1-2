#pragma once

#include <string>
#include <thread>
#include "../header/Barrier.h"

using namespace std;

class ConvolutionRows {
private:
    int N;
    int M;
    int K;
    int P;
    int **matrix;
    int **convolution_matrix;
    thread *threads;

    // clasa noastra barrier
    Barrier &barrier;

    void computeD(int start_idx, int end_idx, int prev_ghost_idx, int next_ghost_idx);


public:
    ConvolutionRows(int N, int M, int K, int P, int **matrix, int **filter,
                    Barrier &barrier);

    void run();

    ~ConvolutionRows();
};