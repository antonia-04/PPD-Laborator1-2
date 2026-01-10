#include "../header/EvaluateTime.h"
#include "../header/ReadFromFile.h"
#include "../header/SequentialConvolution.h"
#include <chrono>

#include "../header/ConvolutionRows.h"
#include "../header/ConvolutionCols.h"
#include <iostream>

#include "../header/DataGeneration.h"

using namespace std::chrono;
using namespace std;

EvaluateTime::EvaluateTime(int N, int M, int P, int K) {
    this->N = N;
    this->K = K;
    this->P = P;
    this->M = M;
    this->matrix = ReadFromFile::readMatrix("matrix.txt");
    this->convolutionMatrix = ReadFromFile::readMatrix("convolutionMatrix.txt");
}

void EvaluateTime::run() {
    cout << "Tip Matrice N=" << N << "; M=" << M << endl;
    cout << "Tip convolutionMatrix: n=m=" << K << endl;
    cout << "Alocare Dinamica" << endl;
    cout << "Secvential: " << estimate_conv_dyn_S() << "ms" << endl;
    cout << "Thread Vertical cu P=" << P << ": " << estimate_conv_dyn_V(P) << "ms" << endl;
    cout << "Thread Orizontal cu P=" << P << ": " << estimate_conv_dyn_H(P) << "ms" << endl;

}

double EvaluateTime::estimate_conv_dyn_S() {

    auto start_time = high_resolution_clock::now();
    SequentialConvolution convolution_s(N, M, K, matrix, convolutionMatrix);
    convolution_s.compute("result.txt");
    auto end_time = high_resolution_clock::now();
    duration<double, milli> round_time = end_time - start_time;
    int **new_matrix = convolution_s.getNewMatrix();
    DataGeneration::writeMatrixToFile(new_matrix, "resultSequentialD.txt", N, M);
    return round_time.count();
}

double EvaluateTime::estimate_conv_dyn_H(const int threads) {
    auto start_time = high_resolution_clock::now();
    ConvolutionRows convolution_h(N, M, K, threads, matrix, convolutionMatrix);
    convolution_h.run();
    auto end_time = high_resolution_clock::now();
    duration<double, milli> round_time = end_time - start_time;
    int **new_matrix = convolution_h.getNewMatrix();
    DataGeneration::writeMatrixToFile(new_matrix, "resultRowsD.txt", N, M);
    return round_time.count();
}

double EvaluateTime::estimate_conv_dyn_V(const int threads) {
    auto start_time = high_resolution_clock::now();
    ConvolutionCols convolution_v(N, M, K, threads, matrix, convolutionMatrix);
    convolution_v.run();
    auto end_time = high_resolution_clock::now();
    duration<double, milli> round_time = end_time - start_time;
    int **new_matrix = convolution_v.getNewMatrix();
    DataGeneration::writeMatrixToFile(new_matrix, "resultColsD.txt", N, M);
    return round_time.count();
}
