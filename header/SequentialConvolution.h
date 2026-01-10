#pragma once

#include <string>
using namespace std;

class SequentialConvolution {
private:
    int N;
    int M;
    int K;
    int **matrix;
    int **convolution_matrix;

public:
    SequentialConvolution(int N, int M, int K, int **matrix, int **filter);

    void compute(const string &result_file);

};