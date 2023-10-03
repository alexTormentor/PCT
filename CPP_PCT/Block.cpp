#define N 10
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stddef.h>
#include "mpi.h"
#include <chrono>
#include <iostream>
#include <ctime>

using namespace std;


void matrix_multiply(int* A, int* B, int* C, int local_rows, int rank) {
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_rows = N / size;
    int* A = new int[N * N];
    int* B = new int[N * N];
    int* C = new int[N * N];
    int* local_A = new int[local_rows * N];
    int* local_C = new int[local_rows * N];


    if (rank == 0) {

        srand(time(NULL));

        for (int i = 0; i < N * N; i++) {
            A[i] = rand() % 10;
        }

        for (int i = 0; i < N * N; i++) {
            B[i] = rand() % 10;
        }
        srand(time(0));
    }

    MPI_Scatter(A, local_rows * N, MPI_INT, local_A, local_rows * N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    matrix_multiply(local_A, B, local_C, local_rows, rank);

    MPI_Gather(local_C, local_rows * N, MPI_INT, C, local_rows * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "runtime = " << clock() / 1000.0 << endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] local_A;
    delete[] local_C;

    MPI_Finalize();
    return 0;
}
