cmake_minimum_required(VERSION 3.27)

# --- CONFIG CUDA (CA ÎN PROIECTUL TESTCUDA) ---
# 1. Spune explicit ce nvcc vrei (CUDA 12.4)
set(CMAKE_CUDA_COMPILER
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe"
)

# 2. Forțează nvcc să accepte MSVC 2026 (19.50)
set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler"
)

# --- PROIECTUL TĂU ---
project(PPD_Laborator1_C__ LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(PPD_Laborator1_C__
        main.cpp
        source/DataGeneration.cpp
        header/DataGeneration.h
        source/ReadFromFile.cpp
        header/ReadFromFile.h
        source/SequentialConvolution.cpp
        header/SequentialConvolution.h
        source/ConvolutionRows.cpp
        header/ConvolutionRows.h
        source/EvaluateTime.cpp
        header/EvaluateTime.h
        header/Barrier.h
        header/ConvolutionCUDABuffer.h
        # IMPORTANT: fișierul CUDA trebuie să fie .cu
        source/ConvolutionCUDABuffer.cu
        header/ConvolutionCUDASharedM.h
        source/ConvolutionCUDASharedM.cu
        header/ConvolutionCUDAClassic.h
        source/ConvolutionCUDAClassic.cu
        source/WarmUpCUDA.cu
)

# Opțional, dar util: spune CMake-ului să facă separable compilation pe CUDA
set_target_properties(PPD_Laborator1_C__ PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

if (MINGW)
    target_link_options(PPD_Laborator1_C__ PRIVATE -static -static-libgcc -static-libstdc++)
endif()
