#pragma once

#include <string>

// clasa care implementeaza convolutia pe gpu folosind shared memory si tiling pe blocuri
// matricea este modificata in-place pe host (la final contine imaginea filtrata)
class ConvolutionCUDASharedM {
private:
    int N;                     // numarul de linii ale matricii de intrare
    int M;                     // numarul de coloane ale matricii de intrare
    int K;                     // dimensiunea kernelului (k x k), in acest proiect k = 3
    int **matrix;              // matricea de pe host (n x m), modificata in-place
    int **convolution_matrix;  // matricea de filtrare (k x k) pe host, aici k = 3

public:
    ConvolutionCUDASharedM(int N, int M, int K, int **matrix, int **filter);

    // calculeaza convolutia in-place pe 'matrix' folosind tiling 2d si shared memory
    // parametrul result_file este pastrat doar pentru simetrie cu celelalte clase, nu este folosit
    void compute(const std::string &result_file);
};
