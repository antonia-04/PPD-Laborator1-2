#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;
using namespace std::chrono;

void writeMatrixToFile(const vector<vector<int>> &matrix, const string &matrix_file) {
    ofstream outM(matrix_file);
    int rows = matrix.size();
    int cols = matrix[0].size();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outM << matrix[i][j] << " ";
        }
        outM << endl;
    }
}

void readMatrix(const string &file_name, vector<vector<int>> &matrix) {
    ifstream in(file_name);
    int rows, cols;
    in >> rows >> cols;
    matrix.resize(rows, vector<int>(cols));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            in >> matrix[i][j];
}

void readfilter(const string &file_name, vector<vector<int>> &filter) {
    ifstream in(file_name);
    int rows, cols;
    in >> rows >> cols;
    filter.resize(rows, vector<int>(cols));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            in >> filter[i][j];
}

int compute_element(int i, int j, const vector<vector<int>> &matrix, const vector<vector<int>> &filter) {
    int N = matrix.size();
    int M = matrix[0].size();
    int K = filter.size();
    int half = K / 2;
    int sum = 0;

    for (int a = -half; a <= half; a++) {
        for (int b = -half; b <= half; b++) {
            int x = i + a;
            int y = j + b;
            if (x < 0) x = 0;
            if (y < 0) y = 0;
            if (x >= N) x = N - 1;
            if (y >= M) y = M - 1;
            sum += matrix[x][y] * filter[a + half][b + half];
        }
    }
    return sum;
}

void runS(const vector<vector<int>> &matrix, const vector<vector<int>> &filter, vector<vector<int>> &new_matrix) {
    int N = matrix.size();
    int M = matrix[0].size();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            new_matrix[i][j] = compute_element(i, j, matrix, filter);
}

void calculate_convolutionV(int start_idx, int end_idx, vector<vector<int>> &new_matrix, const vector<vector<int>> &matrix, const vector<vector<int>> &filter) {
    int N = matrix.size();
    for (int j = start_idx; j < end_idx; j++)
        for (int i = 0; i < N; i++)
            new_matrix[i][j] = compute_element(i, j, matrix, filter);
}

void runV(int P, const vector<vector<int>> &matrix, const vector<vector<int>> &filter, vector<vector<int>> &new_matrix) {
    int M = matrix[0].size();
    vector<thread> threads(P);
    int cols_per_thread = M / P;
    int extra = M % P;
    int start_idx = 0;

    for (int i = 0; i < P; i++) {
        int end_idx = start_idx + cols_per_thread + (extra > 0 ? 1 : 0);
        if (extra > 0) extra--;
        threads[i] = thread(calculate_convolutionV, start_idx, end_idx, ref(new_matrix), cref(matrix), cref(filter));
        start_idx = end_idx;
    }
    for (auto &t : threads) t.join();
}

void calculate_convolutionH(int start_idx, int end_idx, vector<vector<int>> &new_matrix, const vector<vector<int>> &matrix, const vector<vector<int>> &filter) {
    int M = matrix[0].size();
    for (int i = start_idx; i < end_idx; i++)
        for (int j = 0; j < M; j++)
            new_matrix[i][j] = compute_element(i, j, matrix, filter);
}

void runH(int P, const vector<vector<int>> &matrix, const vector<vector<int>> &filter, vector<vector<int>> &new_matrix) {
    int N = matrix.size();
    vector<thread> threads(P);
    int rows_per_thread = N / P;
    int extra = N % P;
    int start_idx = 0;

    for (int i = 0; i < P; i++) {
        int end_idx = start_idx + rows_per_thread + (extra > 0 ? 1 : 0);
        if (extra > 0) extra--;
        threads[i] = thread(calculate_convolutionH, start_idx, end_idx, ref(new_matrix), cref(matrix), cref(filter));
        start_idx = end_idx;
    }
    for (auto &t : threads) t.join();
}

double estimate_conv_S(const vector<vector<int>> &matrix, const vector<vector<int>> &filter) {
    vector<vector<int>> new_matrix(matrix.size(), vector<int>(matrix[0].size()));
    auto start_time = high_resolution_clock::now();
    runS(matrix, filter, new_matrix);
    auto end_time = high_resolution_clock::now();
    writeMatrixToFile(new_matrix, "resultS-S.txt");
    return duration<double, milli>(end_time - start_time).count();
}

double estimate_conv_V(int threads, const vector<vector<int>> &matrix, const vector<vector<int>> &filter) {
    vector<vector<int>> new_matrix(matrix.size(), vector<int>(matrix[0].size()));
    auto start_time = high_resolution_clock::now();
    runV(threads, matrix, filter, new_matrix);
    auto end_time = high_resolution_clock::now();
    writeMatrixToFile(new_matrix, "resultV-S.txt");
    return duration<double, milli>(end_time - start_time).count();
}

double estimate_conv_H(int threads, const vector<vector<int>> &matrix, const vector<vector<int>> &filter) {
    vector<vector<int>> new_matrix(matrix.size(), vector<int>(matrix[0].size()));
    auto start_time = high_resolution_clock::now();
    runH(threads, matrix, filter, new_matrix);
    auto end_time = high_resolution_clock::now();
    writeMatrixToFile(new_matrix, "resultH-S.txt");
    return duration<double, milli>(end_time - start_time).count();
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        cout << "Usage: program P\n";
        return 1;
    }
    int P = atoi(argv[1]);

    vector<vector<int>> matrix;
    vector<vector<int>> filter;

    readMatrix("matrix.txt", matrix);
    cout << "Matrix read from file." << endl;
    readfilter("convolutionMatrix.txt", filter);
    cout << "Filter read from file." << endl;

    cout << "Tip Matrice N=" << matrix.size() << "; M=" << matrix[0].size() << endl;
    cout << "Tip convolutionMatrix: n=m=" << filter.size() << endl;

    cout << "Secvential: " << estimate_conv_S(matrix, filter) << " ms" << endl;
    cout << "Thread Vertical cu P=" << P << ": " << estimate_conv_V(P, matrix, filter) << " ms" << endl;
    cout << "Thread Orizontal cu P=" << P << ": " << estimate_conv_H(P, matrix, filter) << " ms" << endl;

    return 0;
}
