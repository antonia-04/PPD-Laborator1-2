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
        source/ConvolutionCols.cpp
        header/ConvolutionCols.h
        source/EvaluateTime.cpp
        header/EvaluateTime.h
)

add_executable(StaticMain
        static.cpp
)

add_executable(StaticBig
        staticBig.cpp
)

if (MINGW)
    target_link_options(PPD_Laborator1_C__ PRIVATE -static -static-libgcc -static-libstdc++)
    target_link_options(StaticBig PRIVATE -static -static-libgcc -static-libstdc++)
    target_link_options(StaticMain PRIVATE -static -static-libgcc -static-libstdc++)

endif()
