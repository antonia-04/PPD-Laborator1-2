#pragma once

#include <string>
using namespace std;

// CUDA-based convolution, in-place, with O(M) auxiliary memory on device.
class ConvolutionCUDA {
private:
    int N;
    int M;
    int K;
    int **matrix;               // host matrix (N rows, each row has M ints)
    int **convolution_matrix;   // host filter matrix (K x K), here K=3

public:
    ConvolutionCUDA(int N, int M, int K, int **matrix, int **filter);

    // Computes the convolution in-place on 'matrix'.
    // 'result_file' is kept for symmetry with SequentialConvolution, but not used here.
    void compute(const string &result_file);
};
