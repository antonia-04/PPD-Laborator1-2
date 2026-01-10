#pragma once
#include <string>
#include <fstream>

using namespace std;

class ReadFromFile {
    int N;
    int M;

public:
    ReadFromFile(int N, int M);

    static int **readMatrix(const string &file_name);

    bool filesAreEqual(const string &file_name1, const string &file_name2);

};