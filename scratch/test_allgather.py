'''
Let each rank calculate a chunk of q_bore and use
allgatherv to create complete q_bore on each rank
'''
import sys
import time
import numpy as np
import qpoint as qp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

size = 1 if not size else size

chunksize = 100003
fsamp = 100
scan_speed = 1
az_throw = 50
scan_period = 2 * az_throw / float(scan_speed) # in deg.
lat =  10
lon = 10

# all ranks init QMap
Q = qp.QMap(pol=True)

ctime = np.arange(chunksize, dtype=float)
ctime /= float(fsamp)
ctime += time.time()

az = np.arange(chunksize, dtype=float)
az *= (2 * np.pi / scan_period / float(fsamp))
np.sin(az, out=az)
np.arcsin(az, out=az)
az *= (az_throw / np.pi)

el = np.zeros(chunksize, dtype=float) * 10.

# all ranks
t0 = time.time()
q_bore = Q.azel2bore(az, el, None, None, lon, lat, ctime)

if rank == 0:
    print time.time() - t0 

sub_size = np.zeros(size, dtype=int)
quot, remainder = np.divmod(chunksize, size)
sub_size += quot
if remainder:
    # give first ranks one extra quaternion
    sub_size[:int(remainder)] += 1

if rank == 0:
    print sub_size 

start = np.sum(sub_size[:rank], dtype=int)
end = start + sub_size[rank] 

t1 = time.time()
q_bore2 = np.empty(chunksize * 4, dtype=float)

q_boresub = Q.azel2bore(az[start:end], el[start:end], None, None, lon, lat, ctime[start:end])
q_boresub = q_boresub.ravel()

if rank == 1:
    time.sleep(1)

sub_size *= 4 # for the flattened quat array

offsets = np.zeros(size) 
offsets[1:] = np.cumsum(sub_size)[:-1] # start * 4 

#comm.Allgather(q_boresub, q_bore2) # needs equal sized arrays on rank
# see https://bitbucket.org/mpi4py/mpi4py/issues/43/issue-python2-python3
# http://www.tharwan.de/MPI4PY-Allgatherv.html

comm.Allgatherv(q_boresub, [q_bore2, sub_size, offsets, MPI.DOUBLE])
q_bore2 = q_bore2.reshape(chunksize, 4)

if rank == 0:
    print time.time() - t1 



time.sleep(1)
if rank == 0:
    print np.max(np.abs(q_bore - q_bore2))
    print q_bore.shape
    print q_bore2.shape
    print np.allclose(q_bore, q_bore2)

    print q_bore.flags
    print q_bore2.flags

    print q_bore2.base 
    print q_bore2.ravel()


    
ra, dec, pa = Q.bore2radec(np.array([1,0,0,0]),
                              ctime,
                              q_bore,
                              q_hwp=None, sindec=False, return_pa=True)

time.sleep(rank*0.2)
print rank, ra
