from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"rank {rank}")
print(f"size: {size}")


if rank == 0:
    print("ayo 0")
    wa = np.array([0, 1, 3], dtype=np.int16)
    ble = np.empty(3, dtype=np.int16)
    # comm.Bcast(wa, root=0)
    comm.Bcast(np.array([4, 2, 44444], dtype=np.int16))
    comm.Bcast(np.array([0, 2, 3], dtype=np.int16))
    comm.Bcast(np.array([1, 2, 3], dtype=np.int16))
    comm.Bcast(np.array([1, 2, 43], dtype=np.int16), root=0)
elif rank == 1:
    print("ayo 1")
    wa = np.zeros(3, dtype=np.int16)
    ble = np.empty(3, dtype=np.int16)

ble = np.empty(3, dtype=np.int16)
comm.Bcast(wa, root=0)
comm.Bcast(ble, root=0)
print(f"{rank}lo: {wa}")
print(f"{rank}bl: {ble}")


weird = np.zeros(3)
if rank == 0:
    weird[0] = 3
elif rank == 1:
    weird[0] = 1
print(f"{rank} weird{weird}")

