#include <iostream>
#include <fstream>
#include "header/ConvolutionRows.h"
#include "header/DataGeneration.h"
#include "header/SequentialConvolution.h"
#include "header/EvaluateTime.h"
#include "header/ReadFromFile.h"

using namespace std;

bool filesAreEqual(const string &file_name1, const string &file_name2) {
    ifstream f1(file_name1);
    ifstream f2(file_name2);
    if (!f1.is_open() || !f2.is_open()) {
        cout << "Error in opening files" << endl;
        return false;
    }
    f1.seekg(0, ios::end);
    f2.seekg(0, ios::end);
    if (f1.tellg() != f2.tellg()) {
        return false;
    }
    f1.seekg(0, ios::beg);
    f2.seekg(0, ios::beg);

    char c1, c2;
    while (f1.get(c1) && f2.get(c2)) {
        if (c1 != c2) {
            return false;
        }
    }
    return true;
}

void deleteMatrix(int** matrix, int rows) {
    if (matrix != nullptr) {
        for (int i = 0; i < rows; i++) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }
}

int main(int argc, char *argv[]) {

    if (argc < 5) {
        cout << "Usage: ./exec P N M K" << endl;
        cout << "Where:" << endl;
        cout << " P = threads for parallel rows" << endl;
        cout << " N, M = matrix dims" << endl;
        cout << " K = convolution kernel size (K=3 in Lab2)" << endl;
        return 1;
    }

    int P = atoi(argv[1]);
    int N = atoi(argv[2]);
    int M = atoi(argv[3]);
    int K = atoi(argv[4]);

    if (K != 3) {
        cout << "WARNING: Lab 2 specifies K = 3 explicitly!" << endl;
    }

    // generate test files
    DataGeneration generator("matrix.txt", "convolutionMatrix.txt", N, M, K);
    generator.generateMatrix();
    generator.generateFilter();

    // read data once
    int** originalMatrix = ReadFromFile::readMatrix("matrix.txt");
    int** convolutionMatrix = ReadFromFile::readMatrix("convolutionMatrix.txt");

    // evaluate times
    EvaluateTime evaluator(N, M, P, K, originalMatrix, convolutionMatrix);
    evaluator.run();

    cout << endl;
    cout << "============ VERIFICARE CORECTITUDINE ============" << endl;

    bool ok_rows = filesAreEqual("resultRows.txt", "resultSequential.txt");
    cout << "Rows vs Sequential:   " << (ok_rows ? "true" : "false") << endl;

    bool ok_cuda_classic = filesAreEqual("resultCudaClassic.txt", "resultSequential.txt");
    cout << "CUDA (classic) vs Sequential: " << (ok_cuda_classic ? "true" : "false") << endl;


    bool ok_cuda = filesAreEqual("resultCuda.txt", "resultSequential.txt");
    cout << "CUDA vs Sequential:   " << (ok_cuda ? "true" : "false") << endl;

    bool ok_cuda_shared = filesAreEqual("resultCudaShared.txt", "resultSequential.txt");
    cout << "CUDA (shared) vs Sequential: " << (ok_cuda_shared ? "true" : "false") << endl;


    cout << "==================================================" << endl;

    deleteMatrix(originalMatrix, N);
    deleteMatrix(convolutionMatrix, K);

    cout << "Pres ENTER to exit..." << endl;
    cin.get();


    return 0;
}
