from mpi4py import MPI
import numpy as np


neighbors = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3, 4],
    3: [1, 2, 4, 5],
    4: [2, 3, 5, 6],
    5: [3, 4, 6],
    6: [4, 5],
}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 1000

np.random.seed(rank)
A = np.random.rand(N, N)
B = np.random.rand(N, N)

C = np.zeros((N, N))

start_time = MPI.Wtime()

for neighbor_rank in neighbors[rank]:
    comm.Send(A, dest=neighbor_rank)

    comm.Recv(B, source=neighbor_rank)

    C += np.dot(A, B)

end_time = MPI.Wtime()

print(f"Process {rank}: Result C =\n{C}")
print(f"Process {rank}: Execution Time = {end_time - start_time} seconds")

if rank == 0:
    all_results = np.zeros((size, N, N))
else:
    all_results = None

comm.Gather(C, all_results, root=0)

if rank == 0:
    final_result = np.sum(all_results, axis=0)
    print("Final Result (Sum of All Matrices):\n", final_result)
