#include <iostream>

#include "header/ConvolutionRows.h"
#include "header/DataGeneration.h"
#include "header/SequentialConvolution.h"
#include "header/ConvolutionCols.h"
#include "header/EvaluateTime.h"
#include "header/ReadFromFile.h"

using namespace std;


const int N = 100;
const int M = 100;
const int K = 3;

bool filesAreEqual(const string &file_name1, const string &file_name2) {
    ifstream f1(file_name1);
    ifstream f2(file_name2);
    if (!f1.is_open() || !f2.is_open()) {
        cout << "Error in opening files" << endl;
        return false;
    }
    f1.seekg(0, ios::end);
    f2.seekg(0, ios::end);
    if (f1.tellg() != f2.tellg()) {
        cout << "Not equal" << endl;
        return false;
    }
    f1.seekg(0, ios::beg);
    f2.seekg(0, ios::beg);

    char c1, c2;
    while (f1.get(c1) && f2.get(c2)) {
        if (c1 != c2) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        cout << "Order is P N M K";
        return 1;
    }
    int P = atoi(argv[1]);
    int N = atoi(argv[2]);
    int M = atoi(argv[3]);
    int K = atoi(argv[4]);
    DataGeneration generator("matrix.txt", "convolutionMatrix.txt", N, M, K);
    generator.generateMatrix();
    generator.generateFilter();

    if (filesAreEqual("resultRowsD.txt", "resultColsD.txt") &&
        filesAreEqual("resultRowsD.txt", "resultSequentialD.txt")) {
        EvaluateTime estimate_time(N, M, P, K);
        estimate_time.run();
    }
}