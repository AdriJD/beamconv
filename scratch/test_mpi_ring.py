'''
Let every rank send a large array to the next rank
'''
import sys
import time
import numpy as np
import qpoint as qp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nside = 512

data = np.ones((10, 12*nside**2), dtype=np.complex128) * rank

print '{} {}'.format(rank, data[0])

#comm.Sendrecv_replace([data, MPI.COMPLEX], dest=np.mod(rank+1, size), sendtag=rank, source=np.mod(rank-1, size), recvtag=np.mod(rank-1, size))
#comm.Sendrecv_replace(data, dest=np.mod(rank+1, size), sendtag=0, source=np.mod(rank-1, size), recvtag=0) #let mpi4py pick type
comm.Sendrecv_replace(data, dest=np.mod(rank+1, size), source=np.mod(rank-1, size)) #tags arent needed


expected = np.ones((10, 12*nside**2), np.complex128) * np.mod(rank - 1, size)
np.testing.assert_array_almost_equal(data, expected)

print 'done'

# now try to sendrecv some python objects
data_obj = dict(rank=rank)
#recv_obj = None

#recv_obj = comm.sendrecv(sendobj=data_obj, dest=np.mod(rank+1, size), recvobj=recv_obj, source=np.mod(rank-1, size))
recv_obj = comm.sendrecv(sendobj=data_obj, dest=np.mod(rank+1, size), source=np.mod(rank-1, size))
assert recv_obj['rank'] == np.mod(rank-1, size)
