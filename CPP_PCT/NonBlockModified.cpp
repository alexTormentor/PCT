#define N 10
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <iostream>


void matrix_multiply(int* A, int* B, int* C, int local_rows) {
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nnodes = 7;
    int index[] = { 3, 5, 8, 9, 11, 13, 15 };
    int edges[] = { 2, 3, 4, 1, 5, 1, 5, 6, 1, 2, 3, 3, 7, 1, 6 };
    int reorder = 0;

    MPI_Comm comm_graph;
    MPI_Graph_create(MPI_COMM_WORLD, nnodes, index, edges, reorder, &comm_graph);

    int* A = new int[N * N];
    int* B = new int[N * N];
    int* C = new int[N * N];
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
        C[i] = 0;
    }


    int local_rows = N / nnodes;

    int* local_A = new int[local_rows * N];
    int* local_C = new int[local_rows * N];
    for (int i = 0; i < local_rows * N; i++) {
        local_C[i] = 0;
    }

    MPI_Scatter(A, local_rows * N, MPI_INT, local_A, local_rows * N, MPI_INT, 0, comm_graph);

    matrix_multiply(local_A, B, local_C, local_rows);

    MPI_Gather(local_C, local_rows * N, MPI_INT, C, local_rows * N, MPI_INT, 0, comm_graph);

    if (rank == 0) {
        std::cout << "Process " << rank << " result matrix C:" << std::endl;
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < N; col++) {
                std::cout << C[row * N + col] << "\t";
            }
            std::cout << std::endl;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] local_A;
    delete[] local_C;
    MPI_Comm_free(&comm_graph);
    MPI_Finalize();

    return 0;
}
