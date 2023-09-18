from mpi4py import MPI
import numpy as np


def matrix_multiply(A, B):
    return np.dot(A, B)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


N = 1000


if rank == 0:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
else:
    A = None
    B = None

A = comm.bcast(A, root=0)
B = comm.bcast(B, root=0)

C = np.zeros((N, N))

# локальное вычисление
local_result = matrix_multiply(A, B)

# параллелим
for i in range(size - 1):
    # вычисление ранга и назначения
    src = (rank - i) % size
    dest = (rank + i) % size

    # отправка/прием матриц по назначению
    comm.Sendrecv_replace(A, dest=dest, source=src)
    comm.Sendrecv_replace(B, dest=dest, source=src)
    comm.barrier()  # синхрон

    # процеесы работают над матрицей
    local_result += matrix_multiply(A, B)

# сбор рез-татов
global_result = comm.gather(local_result, root=0)


if rank == 0:
    final_result = np.sum(global_result, axis=0)
    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)
    print("Resultant Matrix C:")
    print(final_result)
