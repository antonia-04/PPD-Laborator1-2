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

    template<int N, int M>
    static void readMatrixStatic(const string &file_name, int (&matrix)[N][M]) {
        ifstream in(file_name);
        if (!in.is_open()) {
            throw runtime_error("Error File: " + file_name);
        }

        int rows, cols;
        in >> rows >> cols;

        for (int i = 0; i < rows && i < N; i++) {
            for (int j = 0; j < cols && j < M; j++) {
                in >> matrix[i][j];
            }
        }

        in.close();
    }
};


