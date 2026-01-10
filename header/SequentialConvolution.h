#include <string>
using namespace std;

class SequentialConvolution {
private:
    int N;
    int M;
    int K;
    int **matrix;
    int **convolution_matrix;
    int **new_matrix;

public:
    SequentialConvolution(int N, int M, int K, int **matrix, int **filter);

    int compute_element(int i, int jt);


    void compute(const string &result_file);

    int** getNewMatrix();
    ~SequentialConvolution();
};

