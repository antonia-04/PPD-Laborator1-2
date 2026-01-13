#include "../header/ConvolutionCUDA.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

// Simple macro for CUDA error checking
static inline void cudaCheck(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " : "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA failure");
    }
}
// mic utilitar device pentru clamp pe coloane
__device__ inline int clamp_col(int x, int M) {
    if (x < 0) return 0;
    if (x >= M) return M - 1;
    return x;
}

__global__ void convolveRowKernel(int *d_matrix,
                                  const int *rowPrev,
                                  const int *rowCurr,
                                  const int *rowNext,
                                  const int *conv,
                                  int N, int M,
                                  int rowIndex) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;

    int sum = 0;

    int j_left  = clamp_col(j - 1, M);
    int j_mid   = clamp_col(j,     M);
    int j_right = clamp_col(j + 1, M);

    // Filter conv is:
    // [ c0 c1 c2 ]
    // [ c3 c4 c5 ]
    // [ c6 c7 c8 ]

    // upper row
    sum += rowPrev[j_left]  * conv[0];
    sum += rowPrev[j_mid]   * conv[1];
    sum += rowPrev[j_right] * conv[2];

    // middle row
    sum += rowCurr[j_left]  * conv[3];
    sum += rowCurr[j_mid]   * conv[4];
    sum += rowCurr[j_right] * conv[5];

    // bottom row
    sum += rowNext[j_left]  * conv[6];
    sum += rowNext[j_mid]   * conv[7];
    sum += rowNext[j_right] * conv[8];

    // write result in-place
    d_matrix[rowIndex * M + j] = sum;
}


// -------------------------
// ConvolutionCUDA methods
// -------------------------

ConvolutionCUDA::ConvolutionCUDA(int N, int M, int K, int **matrix, int **filter) {
    this->N = N;
    this->M = M;
    this->K = K;
    this->matrix = matrix;
    this->convolution_matrix = filter;
}

void ConvolutionCUDA::compute(const string & /*result_file*/) {
    if (K != 3) {
        throw std::runtime_error("ConvolutionCUDA currently supports only K=3.");
    }

    const int totalSize = N * M;

    // 1. Flatten host matrix (int** -> int*) onto device
    int *d_matrix = nullptr;
    cudaCheck(cudaMalloc(&d_matrix, totalSize * sizeof(int)), "malloc d_matrix");

    // Copy row by row because matrix is int** with separate allocations per row.
    for (int i = 0; i < N; ++i) {
        cudaCheck(cudaMemcpy(d_matrix + i * M, matrix[i],
                             M * sizeof(int), cudaMemcpyHostToDevice),
                  "memcpy host->device row");
    }

    // 2. Prepare and copy convolution filter (3x3) to device
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

    // 3. Allocate device buffers for 3 rows: prev, curr, next (O(M) memory)
    int *d_rowPrev = nullptr;
    int *d_rowCurr = nullptr;
    int *d_rowNext = nullptr;

    cudaCheck(cudaMalloc(&d_rowPrev, M * sizeof(int)), "malloc d_rowPrev");
    cudaCheck(cudaMalloc(&d_rowCurr, M * sizeof(int)), "malloc d_rowCurr");
    cudaCheck(cudaMalloc(&d_rowNext, M * sizeof(int)), "malloc d_rowNext");

    // 4. Initialize the "sliding window" of 3 rows (similar to SequentialConvolution)
    // For rowIndex = 0:
    //  prevRow: row 0 (clamping -1 -> 0)
    //  currRow: row 0
    //  nextRow: row 1 (or 0 if N == 1)
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

    // 5. Configure kernel launch (1D over columns)
    const int blockSize = 256;
    const int gridSize  = (M + blockSize - 1) / blockSize;

    // 6. Process each row sequentially in terms of rows,
    //    but each row is processed in parallel over columns on the GPU.
    for (int i = 0; i < N; ++i) {
        // Launch kernel for row i
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

        // Prepare sliding window for the next row
        if (i < N - 1) {
            // Rotate buffers: prev <- curr, curr <- next, next <- prev
            int *tmp = d_rowPrev;
            d_rowPrev = d_rowCurr;
            d_rowCurr = d_rowNext;
            d_rowNext = tmp;

            // Next "ghost" row index: i+2 (clamped to N-1)
            int nextIndex = (i + 2 < N) ? (i + 2) : (N - 1);

            // Load original row (i+2 or N-1) into d_rowNext
            // Note: rows above current 'i' may already be overwritten in d_matrix,
            // but rows >= i+1 are still original at this point.
            cudaCheck(cudaMemcpy(d_rowNext, d_matrix + nextIndex * M,
                                 M * sizeof(int), cudaMemcpyDeviceToDevice),
                      "update d_rowNext");
        }
    }

    // 7. Copy back the result from device to host matrix (in-place final state)
    for (int i = 0; i < N; ++i) {
        cudaCheck(cudaMemcpy(matrix[i], d_matrix + i * M,
                             M * sizeof(int), cudaMemcpyDeviceToHost),
                  "memcpy device->host row");
    }

    // 8. Free device memory
    cudaFree(d_matrix);
    cudaFree(d_conv);
    cudaFree(d_rowPrev);
    cudaFree(d_rowCurr);
    cudaFree(d_rowNext);
}
