#include "../header/ConvolutionCUDASharedM.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

using std::string;

// dimensiunea blocului 2d (block_dim x block_dim), folosita atat pentru grid cat si pentru shared memory
#define BLOCK_DIM 16

// macro pentru accesarea colturilor din bufferCorners
// bufferCorners este indexat ca [by][bx][4], iar c indica ce colt este:
// c = 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
#define CORNER_AT(buffer, bx, by, gridX, c) buffer[(((by) * (gridX) + (bx)) * 4) + (c)]


static inline void cudaCheck(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " : "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA failure");
    }
}

// ------------------------ KERNELS ----------------------------

// kernel care salveaza valorile de pe bordurile blocurilor in buffere auxiliare
// - doua randuri per bloc (sus / jos) in bufferRows
// - doua coloane per bloc (stanga / dreapta) in bufferColumns
// - cele 4 colturi per bloc in bufferCorners
// aceste buffere sunt folosite ulterior pentru a calcula pixeli ai caror vecini se afla in blocuri vecine
__global__
void saveBordersKernel(int* matrix,
                       int* bufferColumns,
                       int* bufferRows,
                       int* bufferCorners,
                       int N, int M,
                       int gridX)
{
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    // coordonatele globale ale elementului procesat de acest thread
    int col = blockIdx.x * blockDim.x + threadX;
    int row = blockIdx.y * blockDim.y + threadY;

    if (row >= N || col >= M) return;

    // salvam randurile blocului curent in bufferRows
    if (threadY == 0) {
        // randul de sus al blocului (sau primul rand al matricii)
        bufferRows[(2 * blockIdx.y) * M + col] = matrix[row * M + col];
    }
    if (threadY == blockDim.y - 1 || row == N - 1) {
        // randul de jos al blocului (sau ultimul rand al matricii)
        bufferRows[(2 * blockIdx.y + 1) * M + col] = matrix[row * M + col];
    }

    // salvam coloanele blocului curent in bufferColumns
    if (threadX == 0) {
        // coloana din stanga a blocului
        bufferColumns[(2 * blockIdx.x) * N + row] = matrix[row * M + col];
    }

    if (threadX == blockDim.x - 1 || col == M - 1) {
        // coloana din dreapta a blocului
        bufferColumns[(2 * blockIdx.x + 1) * N + row] = matrix[row * M + col];
    }

    // salvam colturile blocului curent in bufferCorners
    if (threadX == 0 && threadY == 0) {
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // coordonatele colturilor blocului, clamplate la marginea matricii
        int row0 = by * blockDim.y;
        int col0 = bx * blockDim.x;
        int row1 = min(row0 + (int)blockDim.y - 1, N - 1);
        int col1 = min(col0 + (int)blockDim.x - 1, M - 1);

        // se salveaza cele 4 colturi in bufferCorners
        CORNER_AT(bufferCorners, bx, by, gridX, 0) = matrix[row0 * M + col0]; // top-left
        CORNER_AT(bufferCorners, bx, by, gridX, 1) = matrix[row0 * M + col1]; // top-right
        CORNER_AT(bufferCorners, bx, by, gridX, 2) = matrix[row1 * M + col0]; // bottom-left
        CORNER_AT(bufferCorners, bx, by, gridX, 3) = matrix[row1 * M + col1]; // bottom-right
    }
}

// kernel principal de convolutie 3x3
// foloseste:
// - tile 2d (block_dim x block_dim) incarcat in shared memory pentru interiorul blocului
// - bufferele bufferRows / bufferColumns / bufferCorners pentru vecinii care se afla in blocurile vecine
__global__
void calculateKernelShared(int *matrix,
                           int n, int m,
                           int *kernel,
                           int* bufferColumns,
                           int* bufferRows,
                           int* bufferCorners,
                           int gridX)
{
    // tile-ul local din bloc salvat in shared memory
    __shared__ int shared_block[BLOCK_DIM][BLOCK_DIM];

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    // coordonatele globale ale elementului procesat de acest thread
    int col = blockIdx.x * blockDim.x + threadX;
    int row = blockIdx.y * blockDim.y + threadY;

    // incarcam tile-ul curent din matricea globala in shared memory
    if (row < n && col < m) {
        shared_block[threadY][threadX] = matrix[row * m + col];
    } else {
        // daca suntem in afara matricei, completam cu 0 (nu contribuie la suma)
        shared_block[threadY][threadX] = 0;
    }

    __syncthreads();

    if (row < n && col < m) {
        int suma = 0;

        // parcurgem masca 3x3 in jurul pixelului (row, col)
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int val = 0;

                int neighRow = row + di;
                int neighCol = col + dj;

                // aplicam clamping global pentru replicarea marginilor matricii
                int readRow = neighRow;
                int readCol = neighCol;

                if (readRow < 0)      readRow = 0;
                if (readRow >= n)     readRow = n - 1;
                if (readCol < 0)      readCol = 0;
                if (readCol >= m)     readCol = m - 1;

                // calculam coordonatele locale in interiorul tile-ului
                int localRow = threadY + (readRow - row);
                int localCol = threadX + (readCol - col);

                if (localRow >= 0 && localRow < BLOCK_DIM &&
                    localCol >= 0 && localCol < BLOCK_DIM) {
                    // daca vecinul se afla in interiorul aceluiasi bloc, luam valoarea din shared memory
                    val = shared_block[localRow][localCol];
                } else {
                    // daca vecinul se afla in alt bloc, citim din bufferele de bordura
                    int bx = blockIdx.x;
                    int by = blockIdx.y;

                    // cazuri pentru vecini situati pe colturile blocurilor (blocuri diagonale)
                    if (localRow < 0 && localCol < 0) {
                        // bloc sus-stanga
                        if (bx > 0 && by > 0)
                            val = CORNER_AT(bufferCorners, bx - 1, by - 1, gridX, 3);
                    }
                    else if (localRow < 0 && localCol >= BLOCK_DIM) {
                        // bloc sus-dreapta
                        if (bx < gridDim.x - 1 && by > 0)
                            val = CORNER_AT(bufferCorners, bx + 1, by - 1, gridX, 2);
                    }
                    else if (localRow >= BLOCK_DIM && localCol < 0) {
                        // bloc jos-stanga
                        if (bx > 0 && by < gridDim.y - 1)
                            val = CORNER_AT(bufferCorners, bx - 1, by + 1, gridX, 1);
                    }
                    else if (localRow >= BLOCK_DIM && localCol >= BLOCK_DIM) {
                        // bloc jos-dreapta
                        if (bx < gridDim.x - 1 && by < gridDim.y - 1)
                            val = CORNER_AT(bufferCorners, bx + 1, by + 1, gridX, 0);
                    }
                    // cazuri pentru vecini doar pe directia verticala (sus / jos)
                    else if (localRow < 0) { // vecin sus
                        if (by > 0)
                            val = bufferRows[(2 * (by - 1) + 1) * m + readCol];
                    }
                    else if (localRow >= BLOCK_DIM) { // vecin jos
                        if (by < gridDim.y - 1)
                            val = bufferRows[(2 * (by + 1)) * m + readCol];
                    }
                    // cazuri pentru vecini doar pe directia orizontala (stanga / dreapta)
                    else if (localCol < 0) { // vecin stanga
                        if (bx > 0)
                            val = bufferColumns[(2 * (bx - 1) + 1) * n + readRow];
                    }
                    else if (localCol >= BLOCK_DIM) { // vecin dreapta
                        if (bx < gridDim.x - 1)
                            val = bufferColumns[(2 * (bx + 1)) * n + readRow];
                    }
                }

                // kernelul este 3x3, memorat liniar in ordine row-major
                // indexul in kernel se calculeaza din di, dj
                suma += val * kernel[(di + 1) * 3 + (dj + 1)];
            }
        }

        // scriem rezultatul convolutiei in-place in matricea globala pe pozitia (row, col)
        matrix[row * m + col] = suma;
    }
}

ConvolutionCUDASharedM::ConvolutionCUDASharedM(int N, int M, int K,
                                               int **matrix,
                                               int **filter) {
    // salvam dimensiunile si pointerii catre matricea de intrare si filtrul de convolutie
    this->N = N;
    this->M = M;
    this->K = K;
    this->matrix = matrix;
    this->convolution_matrix = filter;
}

void ConvolutionCUDASharedM::compute(const string & /*result_file*/) {
    if (K != 3) {
        throw std::runtime_error("ConvolutionCUDASharedM supports only K=3.");
    }

    const int totalSize = N * M;

    // aplatizam matricea de pe host (int**) intr-un buffer liniar pe device (int*)
    int *d_matrix = nullptr;
    cudaCheck(cudaMalloc(&d_matrix, totalSize * sizeof(int)), "malloc d_matrix");

    // copiem fiecare linie din matricea de pe host in bufferul global de pe device
    for (int i = 0; i < N; ++i) {
        cudaCheck(cudaMemcpy(d_matrix + i * M, matrix[i],
                             M * sizeof(int), cudaMemcpyHostToDevice),
                  "memcpy host->device row");
    }

    // pregatim kernelul 3x3 intr-un vector de 9 elemente pe host si il copiem pe device
    int h_conv[9];
    int idx = 0;
    for (int a = 0; a < K; ++a) {
        for (int b = 0; b < K; ++b) {
            h_conv[idx++] = convolution_matrix[a][b];
        }
    }

    int *d_conv = nullptr;
    cudaCheck(cudaMalloc(&d_conv, 9 * sizeof(int)), "malloc d_conv");
    cudaCheck(cudaMemcpy(d_conv, h_conv, 9 * sizeof(int),
                         cudaMemcpyHostToDevice),
              "memcpy conv host->device");

    // configuram dimensiunile gridului si blocurilor
    // gridX = numarul de blocuri pe orizontala, gridY = numarul de blocuri pe verticala
    int gridX = (M + BLOCK_DIM - 1) / BLOCK_DIM;
    int gridY = (N + BLOCK_DIM - 1) / BLOCK_DIM;

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(gridX, gridY);

    // alocam bufferele pentru borduri:
    // - bufferColumns: cate doua coloane per bloc pe orizontala (stanga/dreapta), pentru toate liniile
    // - bufferRows: cate doua randuri per bloc pe verticala (sus/jos), pentru toate coloanele
    // - bufferCorners: cate 4 colturi pentru fiecare bloc din grid
    int *d_bufferColumns = nullptr;
    int *d_bufferRows    = nullptr;
    int *d_bufferCorners = nullptr;

    // 2 coloane per blocX, fiecare cu n elemente (toate randurile)
    cudaCheck(cudaMalloc(&d_bufferColumns,
                         2 * gridX * N * sizeof(int)),
              "malloc d_bufferColumns");

    // 2 randuri per blocY, fiecare cu m elemente (toate coloanele)
    cudaCheck(cudaMalloc(&d_bufferRows,
                         2 * gridY * M * sizeof(int)),
              "malloc d_bufferRows");

    // 4 colturi per bloc (gridX * gridY * 4 elemente)
    cudaCheck(cudaMalloc(&d_bufferCorners,
                         gridX * gridY * 4 * sizeof(int)),
              "malloc d_bufferCorners");

    // rulam kernelul care salveaza bordurile blocurilor, folosind valorile originale din d_matrix
    // acest pas pregateste informatia necesara pentru a calcula corect vecinii aflati in blocurile vecine
    saveBordersKernel<<<grid, block>>>(d_matrix,
                                       d_bufferColumns,
                                       d_bufferRows,
                                       d_bufferCorners,
                                       N, M, gridX);
    cudaCheck(cudaGetLastError(), "saveBordersKernel launch");
    cudaCheck(cudaDeviceSynchronize(), "saveBordersKernel sync");

    // rulam kernelul principal de convolutie care foloseste shared memory si bufferele de borduri
    // fiecare bloc incarca un tile in shared memory si combina informatia cu bordurile din blocurile vecine
    calculateKernelShared<<<grid, block>>>(d_matrix,
                                           N, M,
                                           d_conv,
                                           d_bufferColumns,
                                           d_bufferRows,
                                           d_bufferCorners,
                                           gridX);
    cudaCheck(cudaGetLastError(), "calculateKernelShared launch");
    cudaCheck(cudaDeviceSynchronize(), "calculateKernelShared sync");

    // copiem rezultatul inapoi in matricea de pe host (in-place)
    // matricea initiala pe host va contine la final imaginea filtrata
    for (int i = 0; i < N; ++i) {
        cudaCheck(cudaMemcpy(matrix[i], d_matrix + i * M,
                             M * sizeof(int), cudaMemcpyDeviceToHost),
                  "memcpy device->host row");
    }

    // eliberam memoria de pe device
    cudaFree(d_matrix);
    cudaFree(d_conv);
    cudaFree(d_bufferColumns);
    cudaFree(d_bufferRows);
    cudaFree(d_bufferCorners);
}
