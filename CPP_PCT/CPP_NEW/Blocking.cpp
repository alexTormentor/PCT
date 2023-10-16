#include <iostream>
#include <vector>
#include "mpi.h"
#include <chrono>  // For measuring time
#include <thread>  // For sleep

using namespace std;

void matrixMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    int N = A.size();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 2;  // Change N to your desired matrix size

    vector<vector<int>> A(N, vector<int>(N, rank + 1));
    vector<vector<int>> B(N, vector<int>(N, rank + 2));
    vector<vector<int>> C(N, vector<int>(N, 0));

    // Create the new communicator using the graph information
    int nnodes = 8;
    int index[8] = { 2, 6, 9, 12, 14, 17, 19, 22 };
    int edges[22] = { 1,7, 0, 2, 3, 4, 1, 4, 5, 1, 5, 6, 1, 2, 2, 3, 7, 3, 7, 0, 6, 5 };
    int reorder = 1;
    MPI_Comm new_comm;
    MPI_Graph_create(MPI_COMM_WORLD, nnodes, index, edges, reorder, &new_comm);

    // Measure the start time
    auto start_time = chrono::high_resolution_clock::now();

    // Perform matrix multiplication
    matrixMultiply(A, B, C);

    // Perform explicit communication based on your desired pattern
    if (rank == 0) {
        // Perform matrix multiplication
        matrixMultiply(A, B, C);
        // Send to process 1 and 3
        MPI_Send(&C[0][0], N * N, MPI_INT, 1, 0, new_comm);
        MPI_Send(&C[0][0], N * N, MPI_INT, 7, 0, new_comm);
    }
    else if (rank == 1) {
        // Receive from process 0 and send to process 2
        MPI_Recv(&C[0][0], N * N, MPI_INT, 0, 0, new_comm, MPI_STATUS_IGNORE);
        // Perform matrix multiplication
        matrixMultiply(A, B, C);
        MPI_Send(&C[0][0], N * N, MPI_INT, 2, 0, new_comm);
        MPI_Send(&C[0][0], N * N, MPI_INT, 3, 0, new_comm);
        MPI_Send(&C[0][0], N * N, MPI_INT, 4, 0, new_comm);
    }
    else if (rank == 2) {
        // Receive from process 3
        MPI_Recv(&C[0][0], N * N, MPI_INT, 1, 0, new_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&C[0][0], N * N, MPI_INT, 4, 0, new_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&C[0][0], N * N, MPI_INT, 5, 0, new_comm, MPI_STATUS_IGNORE);
        // Perform matrix multiplication
        matrixMultiply(A, B, C);
    }
    else if (rank == 3) {
        // Receive from process 0
        MPI_Recv(&C[0][0], N * N, MPI_INT, 1, 0, new_comm, MPI_STATUS_IGNORE);
        // Perform matrix multiplication
        matrixMultiply(A, B, C);
        MPI_Send(&C[0][0], N * N, MPI_INT, 5, 0, new_comm);
        MPI_Send(&C[0][0], N * N, MPI_INT, 6, 0, new_comm);
    }
    else if (rank == 4) {
        // Receive from process 3
        MPI_Recv(&C[0][0], N * N, MPI_INT, 1, 0, new_comm, MPI_STATUS_IGNORE);
        // Perform matrix multiplication
        matrixMultiply(A, B, C);
        MPI_Send(&C[0][0], N * N, MPI_INT, 2, 0, new_comm);
    }
    else if (rank == 5) {
        // Receive from process 3
        MPI_Recv(&C[0][0], N * N, MPI_INT, 3, 0, new_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&C[0][0], N * N, MPI_INT, 7, 0, new_comm, MPI_STATUS_IGNORE);
        // Perform matrix multiplication
        matrixMultiply(A, B, C);
        MPI_Send(&C[0][0], N * N, MPI_INT, 2, 0, new_comm);
    }
    else if (rank == 6) {
        // Receive from process 3
        MPI_Recv(&C[0][0], N * N, MPI_INT, 3, 0, new_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&C[0][0], N * N, MPI_INT, 7, 0, new_comm, MPI_STATUS_IGNORE);
        // Perform matrix multiplication
        matrixMultiply(A, B, C);
    }
    else if (rank == 7) {
        // Receive from process 3
        MPI_Recv(&C[0][0], N * N, MPI_INT, 0, 0, new_comm, MPI_STATUS_IGNORE);
        // Perform matrix multiplication
        matrixMultiply(A, B, C);
        MPI_Send(&C[0][0], N * N, MPI_INT, 5, 0, new_comm);
        MPI_Send(&C[0][0], N * N, MPI_INT, 6, 0, new_comm);
    }



    // Print the result on each process
    cout << "Process " << rank << " result:" << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << C[i][j] << " ";
        }
        cout << endl;
    }


    MPI_Finalize();
    return 0;
}