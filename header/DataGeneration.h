#pragma once
#include <string>

using namespace std;

class DataGeneration {
private:
    string matrix_file;
    string convolution_file;
    int rows;
    int cols;
    int k;

public:

    DataGeneration(const string &matrix_file, const string &filter_file, int n, int m, int k);

    void generateMatrix() const;

    void generateFilter() const;

    static void writeMatrixToFile(int **matrix, const string &matrix_file, const int rows, const int cols);
};