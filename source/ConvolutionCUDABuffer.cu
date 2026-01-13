#include "../header/ConvolutionCUDABuffer.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

// functie auxiliara pentru verificarea codurilor de eroare intoarse de apelurile cuda
// daca apare o eroare, se afiseaza un mesaj si se arunca o exceptie standard
static inline void cudaCheck(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " : "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA failure");
    }
}

// functie helper pe device pentru a limita indexul unei coloane in intervalul [0, m-1]
// aceasta asigura comportamentul de replicare a marginilor pe orizontala
__device__ inline int clamp_col(int x, int M) {
    if (x < 0) return 0;
    if (x >= M) return M - 1;
    return x;
}

// kernel care calculeaza convolutia pentru o singura linie din matrice
// fiecare thread de pe gpu proceseaza un singur element de pe coloana j a liniei rowIndex
// sunt folosite trei linii tampon (rowPrev, rowCurr, rowNext) care contin valori originale
__global__ void convolveRowKernel(int *d_matrix,
                                  const int *rowPrev,
                                  const int *rowCurr,
                                  const int *rowNext,
                                  const int *conv,
                                  int N, int M,
                                  int rowIndex) {
    // j este indexul de coloana procesat de acest thread
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;

    int sum = 0;

    // se calculeaza indicii vecinilor pe orizontala, cu clamp pentru margini
    int j_left  = clamp_col(j - 1, M);
    int j_mid   = clamp_col(j,     M);
    int j_right = clamp_col(j + 1, M);

    // filtrul conv este reprezentat liniar astfel:
    // [ c0 c1 c2 ]
    // [ c3 c4 c5 ]
    // [ c6 c7 c8 ]

    // contributia liniei de sus
    sum += rowPrev[j_left]  * conv[0];
    sum += rowPrev[j_mid]   * conv[1];
    sum += rowPrev[j_right] * conv[2];

    // contributia liniei curente
    sum += rowCurr[j_left]  * conv[3];
    sum += rowCurr[j_mid]   * conv[4];
    sum += rowCurr[j_right] * conv[5];

    // contributia liniei de jos
    sum += rowNext[j_left]  * conv[6];
    sum += rowNext[j_mid]   * conv[7];
    sum += rowNext[j_right] * conv[8];

    // scrie rezultatul in mod in-place in matricea de pe device, pe pozitia corespunzatoare (rowIndex, j)
    d_matrix[rowIndex * M + j] = sum;
}

ConvolutionCUDABuffer::ConvolutionCUDABuffer(int N, int M, int K, int **matrix, int **filter) {
    // se salveaza dimensiunile si pointerii catre matricea de intrare si filtrul de convolutie
    this->N = N;
    this->M = M;
    this->K = K;
    this->matrix = matrix;
    this->convolution_matrix = filter;
}

void ConvolutionCUDABuffer::compute() {
    // kernelul 3x3 (k = 3)
    if (K != 3) {
        throw std::runtime_error("ConvolutionCUDA currently supports only K=3.");
    }

    const int totalSize = N * M;

    // flatten pentru matricea de pe host: int** se transforma logic in int* pe device
    // se aloca un buffer liniar pe device care contine toate elementele matricei
    int *d_matrix = nullptr;
    cudaCheck(cudaMalloc(&d_matrix, totalSize * sizeof(int)), "malloc d_matrix");

    // matricea de pe host este alocata ca int** cu linii separate
    // de aceea copiem linie cu linie catre bufferul liniar de pe device
    for (int i = 0; i < N; ++i) {
        cudaCheck(cudaMemcpy(d_matrix + i * M, matrix[i],
                             M * sizeof(int), cudaMemcpyHostToDevice),
                  "memcpy host->device row");
    }

    // pregatim kernelul de convolutie 3x3 intr-un vector de 9 elemente pe host si il copiem pe device
    int h_conv[9];
    int idx = 0;
    for (int a = 0; a < K; ++a) {
        for (int b = 0; b < K; ++b) {
            h_conv[idx++] = convolution_matrix[a][b];
        }
    }

    int *d_conv = nullptr;
    cudaCheck(cudaMalloc(&d_conv, 9 * sizeof(int)), "malloc d_conv");
    cudaCheck(cudaMemcpy(d_conv, h_conv,
                         9 * sizeof(int), cudaMemcpyHostToDevice),
              "memcpy conv host->device");

    // alocam pe device trei buffere de dimensiune m pentru cele trei linii din fereastra de convolutie:
    //    d_rowPrev, d_rowCurr, d_rowNext
    // acestea asigura memorie auxiliara de ordin m (constrangerea o(n) din tema)
    int *d_rowPrev = nullptr;
    int *d_rowCurr = nullptr;
    int *d_rowNext = nullptr;

    cudaCheck(cudaMalloc(&d_rowPrev, M * sizeof(int)), "malloc d_rowPrev");
    cudaCheck(cudaMalloc(&d_rowCurr, M * sizeof(int)), "malloc d_rowCurr");
    cudaCheck(cudaMalloc(&d_rowNext, M * sizeof(int)), "malloc d_rowNext");

    // initializam fereastra glisanta de 3 linii, analog cu implementarea secventiala
    // pentru rowIndex = 0:
    //  - linia anterioara este considerata tot linia 0 (clamp pentru linia -1)
    //  - linia curenta este linia 0
    //  - linia urmatoare este linia 1 (sau tot 0 daca n == 1)
    int row0 = 0;
    int row1 = (N > 1 ? 1 : 0);

    cudaCheck(cudaMemcpy(d_rowPrev, d_matrix + row0 * M,
                         M * sizeof(int), cudaMemcpyDeviceToDevice),
              "init rowPrev");
    cudaCheck(cudaMemcpy(d_rowCurr, d_matrix + row0 * M,
                         M * sizeof(int), cudaMemcpyDeviceToDevice),
              "init rowCurr");
    cudaCheck(cudaMemcpy(d_rowNext, d_matrix + row1 * M,
                         M * sizeof(int), cudaMemcpyDeviceToDevice),
              "init rowNext");

    // configuram lansarea kernelului: organizare 1d pe coloane
    // un bloc contine blockSize threaduri, iar gridSize este numarul de blocuri necesare pentru m coloane
    const int blockSize = 256;
    const int gridSize  = (M + blockSize - 1) / blockSize;

    // parcurgem liniile de la 0 la n - 1 pe host
    // pentru fiecare linie i, lansam un kernel care proceseaza in paralel toate coloanele acelei linii
    // astfel avem paralelizare pe gpu pe coloane, dar avansam secvential pe linii pentru a pastra corectitudinea in-place
    for (int i = 0; i < N; ++i) {
        // lansam kernelul care calculeaza convolutia pentru linia i, folosind cele trei buffere de linii
        convolveRowKernel<<<gridSize, blockSize>>>(
            d_matrix,
            d_rowPrev,
            d_rowCurr,
            d_rowNext,
            d_conv,
            N, M,
            i
        );
        cudaCheck(cudaGetLastError(), "convolveRowKernel launch");
        cudaCheck(cudaDeviceSynchronize(), "convolveRowKernel sync");

        // pregatim fereastra glisanta pentru linia urmatoare
        if (i < N - 1) {
            // rotim bufferele astfel incat:
            //  - linia curenta devine linia anterioara
            //  - linia urmatoare devine linia curenta
            //  - linia anterioara este refolosita ca buffer pentru noua linie urmatoare
            int *tmp = d_rowPrev;
            d_rowPrev = d_rowCurr;
            d_rowCurr = d_rowNext;
            d_rowNext = tmp;

            // calculam indexul urmatoarei linii "fantoma" i+2 (cu clamp la n-1 pentru marginea de jos)
            int nextIndex = (i + 2 < N) ? (i + 2) : (N - 1);

            // incarcam in d_rowNext linia originala nextIndex din d_matrix
            // observatie: liniile cu index mai mic sau egal cu i au fost deja suprascrise,
            // dar liniile cu index cel putin i+1 sunt inca in starea originala la acest moment
            cudaCheck(cudaMemcpy(d_rowNext, d_matrix + nextIndex * M,
                                 M * sizeof(int), cudaMemcpyDeviceToDevice),
                      "update d_rowNext");
        }
    }

    // la final, copiem rezultatul din d_matrix inapoi in matricea de pe host
    // astfel, matricea initiala va contine imaginea filtrata (postconditia din tema)
    for (int i = 0; i < N; ++i) {
        cudaCheck(cudaMemcpy(matrix[i], d_matrix + i * M,
                             M * sizeof(int), cudaMemcpyDeviceToHost),
                  "memcpy device->host row");
    }

    // eliberam memoria alocata pe device pentru matrice, kernel si bufferele de linii
    cudaFree(d_matrix);
    cudaFree(d_conv);
    cudaFree(d_rowPrev);
    cudaFree(d_rowCurr);
    cudaFree(d_rowNext);
}
