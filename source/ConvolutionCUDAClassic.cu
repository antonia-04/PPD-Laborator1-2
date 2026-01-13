#include "../header/ConvolutionCUDAClassic.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

using std::cerr;
using std::endl;

// configurare simpla a dimensiunii blocurilor pe gpu (dimensiune 2d: block_x x block_y)
#define BLOCK_X 16
#define BLOCK_Y 16

static inline void cudaCheck(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << msg << " : "
             << cudaGetErrorString(err) << endl;
        throw std::runtime_error("CUDA failure");
    }
}

// functie helper pe device pentru clamp pe un interval [low, high]
// este folosita pentru a replica marginile imaginii la calculul vecinilor (tratament borduri)
__device__ inline int clamp_int(int x, int low, int high) {
    if (x < low)  return low;
    if (x > high) return high;
    return x;
}

// kernel de convolutie 2d cu kernel 3x3 si matrice de intrare si iesire separate
// fiecare thread calculeaza exact un element al matricei de iesire, pe pozitia (i, j)
__global__
void convolve2D_classic_kernel(const int *d_input,
                               int *d_output,
                               const int *d_kernel,
                               int N, int M)
{
    // j = indexul de coloana
    // i = indexul de linie pentru elementul curent
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row

    // daca threadul iese in afara matricii, nu calculeaza nimic
    if (i >= N || j >= M) return;

    int sum = 0;

    // kernel 3x3, memorat liniar in ordine row-major:
    // [ k0 k1 k2 ]
    // [ k3 k4 k5 ]
    // [ k6 k7 k8 ]
    // se parcurg cei 9 vecini ai pixelului (i, j) si se aduna contributiile lor
    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            // coordonatele vecinului in matricea de intrare
            int ni = clamp_int(i + di, 0, N - 1);
            int nj = clamp_int(j + dj, 0, M - 1);

            // valoarea pixelului vecin
            int pixel = d_input[ni * M + nj];
            // valoarea corespunzatoare din kernel
            int k_val = d_kernel[(di + 1) * 3 + (dj + 1)];

            sum += pixel * k_val;
        }
    }

    // threadul scrie rezultatul convolutiei in matricea de iesire pe pozitia (i, j)
    d_output[i * M + j] = sum;
}


ConvolutionCUDAClassic::ConvolutionCUDAClassic(int N, int M, int K,
                                               int **inputMatrix,
                                               int **filterMatrix) {
    // salvam dimensiunile si pointerii catre matricea de intrare si matricea kernel
    this->N = N;
    this->M = M;
    this->K = K;
    this->inputMatrix = inputMatrix;
    this->convolutionMatrix = filterMatrix;
}

void ConvolutionCUDAClassic::compute(int **resultMatrix) {
    if (K != 3) {
        throw std::runtime_error("ConvolutionCUDAClassic supports only K=3.");
    }

    const int totalSize = N * M;

    // aplatizam matricea de intrare de pe host intr-un buffer liniar pe device (d_input)
    int *d_input  = nullptr;
    int *d_output = nullptr;

    cudaCheck(cudaMalloc(&d_input,  totalSize * sizeof(int)), "malloc d_input");
    cudaCheck(cudaMalloc(&d_output, totalSize * sizeof(int)), "malloc d_output");

    // copiem fiecare linie din matricea de intrare (int**) in bufferul liniar de pe device
    for (int i = 0; i < N; ++i) {
        cudaCheck(cudaMemcpy(d_input + i * M, inputMatrix[i],
                             M * sizeof(int), cudaMemcpyHostToDevice),
                  "memcpy host->device row");
    }

    // aplatizam kernelul 3x3 intr-un vector de 9 elemente pe host si il copiem pe device
    int h_kernel[9];
    int idx = 0;
    for (int a = 0; a < K; ++a) {
        for (int b = 0; b < K; ++b) {
            h_kernel[idx++] = convolutionMatrix[a][b];
        }
    }

    int *d_kernel = nullptr;
    cudaCheck(cudaMalloc(&d_kernel, 9 * sizeof(int)), "malloc d_kernel");
    cudaCheck(cudaMemcpy(d_kernel, h_kernel,
                         9 * sizeof(int), cudaMemcpyHostToDevice),
              "memcpy kernel host->device");

    // configuram gridul si blocurile pentru lansarea kernelului 2d
    // block = dimensiune fixa (block_x x block_y)
    // grid = numarul de blocuri necesare pentru a acoperi toate liniile si coloanele
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((M + BLOCK_X - 1) / BLOCK_X,
              (N + BLOCK_Y - 1) / BLOCK_Y);

    // lansam kernelul de convolutie 2d pe GPU
    // fiecare thread calculeaza un element al matricei de iesire, folosind matricea de intrare d_input si kernelul d_kernel
    convolve2D_classic_kernel<<<grid, block>>>(d_input, d_output, d_kernel, N, M);
    cudaCheck(cudaGetLastError(), "convolve2D_classic_kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "convolve2D_classic_kernel sync");

    // copiem rezultatul din d_output inapoi in matricea rezultat de pe host (resultMatrix)
    // astfel, resultMatrix va contine imaginea filtrata, iar inputMatrix ramane nemodificata
    for (int i = 0; i < N; ++i) {
        cudaCheck(cudaMemcpy(resultMatrix[i], d_output + i * M,
                             M * sizeof(int), cudaMemcpyDeviceToHost),
                  "memcpy device->host row");
    }

    // eliberam memoria alocata
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}
