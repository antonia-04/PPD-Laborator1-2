#include "../header/DataGeneration.h"
#include <string>
#include <fstream>
#include <cstdlib> // Pentru rand()

using namespace std;

DataGeneration::DataGeneration(const string &matrix_file, const string &convolution_file, int n, int m, int k)
        : matrix_file(matrix_file), convolution_file(convolution_file), rows(n), cols(m), k(k) {}


void DataGeneration::generateMatrix() const {
    ofstream outM(matrix_file);
    outM << rows << " " << cols << endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outM << rand() % 100 << " ";
        }
        outM << endl;
    }
}

void DataGeneration::generateFilter() const {
    ofstream outFilter(convolution_file);
    outFilter << k << " " << k << endl;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            outFilter << rand() % 10 << " ";
        }
        outFilter << endl;
    }
}

void DataGeneration::writeMatrixToFile(int **matrix, const string &matrix_file, const int rows, const int cols) {
    ofstream outM(matrix_file);
    outM << rows << " " << cols << endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outM << matrix[i][j] << " ";
        }
        outM << endl;
    }
}