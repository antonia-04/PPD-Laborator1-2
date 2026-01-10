#include <pthread.h>

class EvaluateTime {
private:
    int **matrix;
    int **convolutionMatrix;
    int P;
    int N;
    int M;
    int K;

public:
    EvaluateTime(int N, int M, int P, int K);

    void run();

    double estimate_conv_dyn_S();

    double estimate_conv_dyn_H(int threads);

    double estimate_conv_dyn_V(int threads);
};
