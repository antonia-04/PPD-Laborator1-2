#pragma once

class EvaluateTime {
private:
    int **originalMatrix;
    int **convolutionMatrix;
    int P;
    int N;
    int M;
    int K;

    // functii helper pt clonare si stergere matrici
    int** deepCopyMatrix(int** matrix, int rows, int cols);
    void deleteMatrix(int** matrix, int rows);

public:
    EvaluateTime(int N, int M, int P, int K, int** matrix, int** convMatrix);

    ~EvaluateTime();

    void run();

    double estimate_conv_dyn_S();

    double estimate_conv_dyn_H(int threads);

    double estimate_conv_cuda();
};