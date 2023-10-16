from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the graph topology
nnodes = 8
index = [2, 6, 9, 12, 14, 17, 19, 22]
edges = [1, 7, 0, 2, 3, 4, 1, 4, 5, 1, 5, 6, 1, 2, 2, 3, 7, 3, 7, 0, 6, 5]
reorder = 1

# Create a communicator based on the graph
comm_graph = comm.Create_graph(index, edges, reorder)

# Check if the current process belongs to the graph communicator
if comm_graph != MPI.COMM_NULL:
    graph_rank = comm_graph.Get_rank()

    # Adjust matrix dimensions to ensure even distribution
    matrix_size = (6, 16)  # 6 rows, 16 columns
    A = np.random.rand(matrix_size[0], matrix_size[1])  # Replace with your own matrix data
    B = np.random.rand(matrix_size[1], matrix_size[1])  # Replace with your own matrix data

    # Calculate local matrix dimensions for each process
    local_A_columns = matrix_size[1] // size
    local_B_rows = matrix_size[1] // size

    local_A = np.zeros((matrix_size[0], local_A_columns), dtype=np.float64)
    local_B = np.zeros((local_B_rows, matrix_size[1]), dtype=np.float64)

    comm.Scatter(A, local_A, root=0)
    comm.Scatter(B, local_B, root=0)

    # Perform local matrix multiplication
    local_C = np.dot(local_A, local_B)

    # Collect local results using collective communication
    global_C = np.zeros((matrix_size[0], matrix_size[1]), dtype=np.float64)
    comm_graph.Allreduce(local_C, global_C, op=MPI.SUM)

    # Print the result
    if rank == 0:
        print("Matrix C:")
        print(global_C)

else:
    print(f"Process {rank} does not belong to the graph communicator.")

# Finalize MPI
MPI.Finalize()
