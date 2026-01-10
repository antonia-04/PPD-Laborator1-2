#include <iostream>
#include "header/ConvolutionRows.h" // Modificat
#include "header/DataGeneration.h"
#include "header/SequentialConvolution.h" // Modificat
#include "header/EvaluateTime.h" // Modificat
#include "header/ReadFromFile.h"

using namespace std;

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

// functie pt eliminarea matricilor alocate dinamic
void deleteMatrix(int** matrix, int rows) {
    if (matrix != nullptr) {
        for (int i = 0; i < rows; i++) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }
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

    if (K != 3) {
        cout << "Atentie: Laboratorul 2 specifica K=3." << endl;
    }

    DataGeneration generator("matrix.txt", "convolutionMatrix.txt", N, M, K);
    generator.generateMatrix();
    generator.generateFilter();

    // citim datele o singura data
    int** originalMatrix = ReadFromFile::readMatrix("matrix.txt");
    int** convolutionMatrix = ReadFromFile::readMatrix("convolutionMatrix.txt");

    EvaluateTime estimate_time(N, M, P, K, originalMatrix, convolutionMatrix);
    estimate_time.run();

    // verificam corectitudinea
    cout << "==========================================" << endl;
    bool ok = filesAreEqual("resultRows.txt", "resultSequential.txt");
    cout << "Verificare corectitudine (Rows vs Seq): " << (ok ? "true" : "false") << endl;
    cout << "==========================================" << endl;

    deleteMatrix(originalMatrix, N);
    deleteMatrix(convolutionMatrix, K);

    return 0;
}