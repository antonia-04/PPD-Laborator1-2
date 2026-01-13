#include <cuda_runtime.h>

extern "C" void warmupCuda() {
    int *d_dummy = nullptr;
    cudaMalloc(&d_dummy, sizeof(int));
    cudaFree(d_dummy);
}
