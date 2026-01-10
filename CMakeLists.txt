cmake_minimum_required(VERSION 3.27)
project(PPD_Laborator1_C__)

set(CMAKE_CXX_STANDARD 20)

add_executable(PPD_Laborator1_C__ main.cpp
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
)



if (MINGW)
    target_link_options(PPD_Laborator1_C__ PRIVATE -static -static-libgcc -static-libstdc++)

endif()
