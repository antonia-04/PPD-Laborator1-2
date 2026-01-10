#include <fstream>
#include "../header/ReadFromFile.h"
#include <iostream>

using namespace std;

ReadFromFile::ReadFromFile(const int N, const int M) {
    this->N=N;
    this->M=M;
}

int **ReadFromFile::readMatrix(const string &file_name) {
    ifstream in(file_name);
    if (!in.is_open()) {
        throw runtime_error("Eroare la deschiderea fisierului: " + file_name);
    }
    int rows, cols;
    in >> rows >> cols;

    int **matrix = new int *[rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new int[cols];
        for (int j = 0; j < cols; j++) {
            in >> matrix[i][j];
        }
    }
    in.close(); // E bine sa inchidem fisierul
    return matrix;
}
bool ReadFromFile::filesAreEqual(const string &file_name1, const string &file_name2) {
    ifstream f1(file_name1);
    ifstream f2(file_name2);
    if (!f1.is_open() || !f2.is_open()) {
        cout << "Error in opening files" << endl;
        return false;
    }
    f1.seekg(0, ios::end);
    f2.seekg(0, ios::end);
    if (f1.tellg() != f2.tellg()) {
        cout << "Not equal"<<endl;
        return false;
    }
    f1.seekg(0, ios::beg);
    f2.seekg(0, ios::beg);

    char c1,c2;
    while (f1.get(c1) && f2.get(c2)) {
        if (c1 != c2) {
            return false;
        }
    }
    return true;
}