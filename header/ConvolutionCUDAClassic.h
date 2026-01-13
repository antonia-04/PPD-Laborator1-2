#pragma once

// clasa care implementeaza varianta clasica de convolutie pe gpu (cuda)
// cu matrice rezultat separata (out-of-place)
class ConvolutionCUDAClassic {
    int N;                    // numarul de linii ale matricii de intrare
    int M;                    // numarul de coloane ale matricii de intrare
    int K;                    // dimensiunea kernelului de convolutie (k x k), in proiect k = 3
    int **inputMatrix;        // matricea de intrare pe host (n x m), nu este modificata de algoritm
    int **convolutionMatrix;  // matricea de filtrare pe host (k x k), k = 3

public:
    // constructor care initializeaza dimensiunile si pointerii catre matricea de intrare si filtrul de convolutie
    ConvolutionCUDAClassic(int N, int M, int K,
                           int **inputMatrix,
                           int **filterMatrix);

    // calculeaza convolutia pe gpu in matricea rezultat de pe host
    // resultMatrix trebuie sa fie deja alocata ca matrice n x m (int**)
    // algoritmul este out-of-place: inputMatrix ramane neschimbata, iar resultMatrix primeste valorile filtrate
    void compute(int **resultMatrix);
};
