#define N 4
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stddef.h>
#include "mpi.h"
#include <chrono>
#include <iostream>
#include <vector>
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

    for (int i = 0; i < nnodes; i++) {
        int dest;
        int source;

        MPI_Graph_neighbors_count(comm_graph, rank, &dest);
        MPI_Graph_neighbors_count(comm_graph, i, &source);

        MPI_Status status;
        MPI_Request send_request, recv_request;

        MPI_Send_init(&A[0], N * N, MPI_INT, i, 0, comm_graph, &send_request);
        MPI_Recv_init(&A[0], N * N, MPI_INT, i, 0, comm_graph, &recv_request);

        MPI_Start(&send_request);
        MPI_Start(&recv_request);

        MPI_Wait(&send_request, &status);
        MPI_Wait(&recv_request, &status);
    }

    MPI_Barrier(comm_graph);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    for (int i = 0; i < size; i++) {
        if (i == rank) {
            std::cout << "Process " << rank << " result matrix C:" << std::endl;
            for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                    std::cout << C[row * N + col] << "\t";
                }
                std::cout << std::endl;
            }
        }
        MPI_Barrier(comm_graph);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    MPI_Comm_free(&comm_graph);
    MPI_Finalize();

    return 0;
}
