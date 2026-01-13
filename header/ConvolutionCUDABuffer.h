#pragma once

#include <string>
using namespace std;

// clasa care implementeaza convolutia pe gpu cu cuda
// in-place, folosind memorie auxiliara de ordin m pe device
class ConvolutionCUDABuffer {
private:
    int N;                      // numarul de linii ale matricii de intrare
    int M;                      // numarul de coloane ale matricii de intrare
    int K;                      // dimensiunea kernelului de convolutie (in proiect este k = 3)
    int **matrix;               // matricea de pe host (n linii, fiecare linie are m int-uri)
    int **convolution_matrix;   // matricea de filtrare (k x k) pe host, aici k = 3

public:
    // constructor care initializeaza dimensiunile si referintele catre matricea de intrare si matricea de convolutie
    ConvolutionCUDABuffer(int N, int M, int K, int **matrix, int **filter);

    // calculeaza convolutia in-place pe matricea 'matrix' folosind gpu
    void compute();
};
