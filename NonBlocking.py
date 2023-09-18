from mpi4py import MPI
import numpy as np


N = 3


neighbors = {
    1: [2, 3],
    2: [1, 5],
    3: [1, 4],
    4: [3, 6, 7],
    5: [2, 6],
    6: [4, 5],
    7: [4]
}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# костыль
if size != len(neighbors):
    if rank == 0:
        print("Error: Number of processes does not match the neighbors table.")
    comm.Abort()


A = np.random.rand(N, N)
B = np.random.rand(N, N)
C = np.zeros((N, N))

send_requests = {}
recv_requests = {}

for neighbor in neighbors[rank + 1]:
    send_requests[neighbor] = None
    recv_requests[neighbor] = None


for i in range(N):
    # промежуточные результаты процессами
    local_result = np.dot(A, B)

    # неблокирующая рассылка
    for neighbor in neighbors[rank + 1]:
        send_requests[neighbor] = comm.Isend(local_result, dest=neighbor - 1)
        recv_requests[neighbor] = comm.Irecv(C, source=neighbor - 1)

    # ждем
    for neighbor in neighbors[rank + 1]:
        send_requests[neighbor].Wait()

    # тут был костыль

    for neighbor in neighbors[rank + 1]:
        recv_requests[neighbor].Wait()

if rank == 0:
    print("Matrix multiplication result:")
    print(C)

# Finalize MPI
MPI.Finalize()



'''
еще не является неблокирующей. В полностью  неблокирующей программе вычисления и обмен данными перекрывались бы, позволяя программе выполнять обе задачи 
одновременно, не дожидаясь завершения одной, прежде чем запускать другую. программа по-прежнему ожидает завершения всех неблокирующих отправлений, прежде чем приступить к вычислению, 
а затем ожидает завершения всех неблокирующих сендов, прежде чем завершить цикл. Это ожидание, по сути, является блокирующей программой
'''