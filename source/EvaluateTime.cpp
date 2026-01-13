#include "../header/EvaluateTime.h"
#include "../header/ReadFromFile.h"
#include "../header/SequentialConvolution.h"
#include "../header/ConvolutionCUDA.h"
#include <chrono>
#include "../header/ConvolutionRows.h"
#include <iostream>
#include "../header/DataGeneration.h"
#include "../header/Barrier.h"
#include <cstring>

using namespace std::chrono;
using namespace std;


int** EvaluateTime::deepCopyMatrix(int** matrix, int rows, int cols) {
    if (matrix == nullptr) return nullptr;
    int** copy = new int*[rows];
    for (int i = 0; i < rows; i++) {
        copy[i] = new int[cols];
        memcpy(copy[i], matrix[i], cols * sizeof(int));
    }
    return copy;
}

void EvaluateTime::deleteMatrix(int** matrix, int rows) {
    if (matrix != nullptr) {
        for (int i = 0; i < rows; i++) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }
}



EvaluateTime::EvaluateTime(int N, int M, int P, int K, int** matrix, int** convMatrix) {
    this->N = N;
    this->K = K;
    this->P = P;
    this->M = M;
    this->originalMatrix = deepCopyMatrix(matrix, N, M);
    this->convolutionMatrix = deepCopyMatrix(convMatrix, K, K);
}

EvaluateTime::~EvaluateTime() {
    deleteMatrix(originalMatrix, N);
    deleteMatrix(convolutionMatrix, K);
}

void EvaluateTime::run() {
    cout << "Tip Matrice N=" << N << "; M=" << M << endl;
    cout << "Tip convolutionMatrix: n=m=" << K << endl;
    cout << "Alocare Dinamica (int**)" << endl;
    cout << "Secvential: " << estimate_conv_dyn_S() << "ms" << endl;
    cout << "Thread Orizontal cu P=" << P << ": " << estimate_conv_dyn_H(P) << "ms" << endl;
    double cudaTime = estimate_conv_cuda();
    cout << "Timp CUDA:              " << cudaTime << " ms\n";
}

double EvaluateTime::estimate_conv_dyn_S() {
    int** matrixForSeq = deepCopyMatrix(originalMatrix, N, M);

    auto start_time = high_resolution_clock::now();
    SequentialConvolution convolution_s(N, M, K, matrixForSeq, convolutionMatrix);
    convolution_s.compute("resultSequential.txt");
    auto end_time = high_resolution_clock::now();

    duration<double, milli> round_time = end_time - start_time;
    DataGeneration::writeMatrixToFile(matrixForSeq, "resultSequential.txt", N, M);

    deleteMatrix(matrixForSeq, N);

    return round_time.count();
}

double EvaluateTime::estimate_conv_dyn_H(const int threads) {
    int** matrixForRows = deepCopyMatrix(originalMatrix, N, M);
    Barrier barrier(threads);

    auto start_time = high_resolution_clock::now();

    ConvolutionRows convolution_h(N, M, K, threads, matrixForRows, convolutionMatrix, barrier);
    convolution_h.run();
    auto end_time = high_resolution_clock::now();

    duration<double, milli> round_time = end_time - start_time;
    DataGeneration::writeMatrixToFile(matrixForRows, "resultRows.txt", N, M);

    deleteMatrix(matrixForRows, N);

    return round_time.count();
}

double EvaluateTime::estimate_conv_cuda() {
    using namespace std::chrono;

    // copiem matricea originală ca să nu o stricăm
    int **matrixForCuda = deepCopyMatrix(originalMatrix, N, M);

    auto start_time = high_resolution_clock::now();

    ConvolutionCUDA conv_cuda(N, M, K, matrixForCuda, convolutionMatrix);
    conv_cuda.compute("resultCuda.txt");

    auto end_time = high_resolution_clock::now();
    duration<double, milli> elapsed = end_time - start_time;

    // scriem rezultatul ca să putem compara cu secvențial/paralel
    DataGeneration::writeMatrixToFile(matrixForCuda, "resultCuda.txt", N, M);

    deleteMatrix(matrixForCuda, N);

    return elapsed.count();
}

