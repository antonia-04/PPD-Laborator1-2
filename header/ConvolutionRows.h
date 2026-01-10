#include <string>
#include <thread>
using namespace std;

class ConvolutionRows {
private:
    int N;
    int M;
    int K;
    int P;
    int **matrix;
    int **convolution_matrix;
    int **new_matrix;
    thread *threads;

    void computeD(int start_idx, int end_idx);

    int compute_element(int i, int j);

public:
    ConvolutionRows(int N, int M, int K, int P, int **matrix, int **filter);

    void run();
    int ** getNewMatrix();
};
