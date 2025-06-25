import npmps
from numba import njit, prange
import os
import numpy as np

@njit(parallel=True)
def dec_to_bin(dec, N):
    bstr = np.zeros(N, dtype=np.int32)
    for i in prange(N):
        bstr[N - 1 - i] = (dec >> i) & 1
    return bstr

@njit
def bin_array_to_dec(bstr, rescale=1.0, shift=0.0):
    dec = 0
    for i in range(len(bstr)):
        dec += bstr[i] * (2 ** i)
    return dec * rescale + shift

@njit(parallel=True)
def bin_to_dec_list(bstr, rescale=1.0, shift=0.0):
    dec_list = np.zeros(len(bstr), dtype=np.float64)
    for i in prange(len(bstr)):
        dec_list[i] = bin_array_to_dec(bstr[i], rescale, shift)
    return dec_list

# An iterator for binary numbers
class BinaryNumbers:
    def __init__(self, N):
        self.N_num = N
        self.N_dec = 2**N

    def __iter__(self):
        self.dec = 0
        return self

    def __next__(self):
        if self.dec < self.N_dec:
            dec = self.dec
            self.dec += 1
            return dec_to_bin(dec, self.N_num)[::-1]
        else:
            raise StopIteration

@njit
def get_ele_mps(mps, bstr):
    assert len(mps) == len(bstr)
    res = np.array([[1.]], dtype=mps[0].dtype)
    for i in range(len(mps)):  # Avoid parallelizing this loop for safety
        A = mps[i]  # Ensure dtype is np.float64
        bi = bstr[i]
        M = np.ascontiguousarray(A[:, bi, :].astype(mps[0].dtype))
        res = np.dot(res, M)  # Consider using np.matmul or direct dot product
    return res[0][0]


@njit
def get_ele_mpo(mpo, L, R, bstr):
    mpo = npmps.absort_LR(mpo, L, R)
    res = np.array([[1.0]], dtype=mpo[0].dtype)
    for i in range(len(mpo)):  # Parallelize this loop
        A = mpo[i]
        bi = bstr[i]
        M = np.ascontiguousarray(A[:, bi, bi, :].astype(mpo[0].dtype))
        res = np.dot(res, M)
    return res[0][0]

@njit(parallel=True)
def get_2D_mesh_eles_mps(mps, bxs, bys):
    nx, ny = len(bxs), len(bys)
    total = nx * ny
    fs = np.zeros((ny, nx), dtype=mps[0].dtype)
    bstr_len = len(bxs[0]) + len(bys[0])  
    for idx in prange(total):
        bstr = np.empty(bstr_len, dtype=bxs[0].dtype)
        i = idx % nx
        j = idx // nx
        
        bstr[:len(bxs[i])] = bxs[i]
        bstr[len(bxs[i]):] = bys[j]
        
        fs[j, i] = get_ele_mps(mps, bstr)
    
    return fs

@njit(parallel=True)
def get_3D_mesh_eles_mps(mps, bxs, bys, bzs):
    nx, ny, nz = len(bxs), len(bys), len(bzs)
    total = nx * ny * nz
    fs = np.zeros((nz, ny, nx), dtype=mps[0].dtype)
    
    for idx in prange(total):
        bstr = np.empty(len(bxs[0]) + len(bys[0]) + len(bzs[0]), dtype=bxs[0].dtype)  
        i = idx % nx
        j = (idx // nx) % ny
        k = idx // (nx * ny)

        bstr[:] = np.hstack((bxs[i], bys[j], bzs[k]))
        fs[k, j, i] = get_ele_mps(mps, bstr)
   
    return fs

# Function for 2D mesh MPO
@njit(parallel=True)
def get_2D_mesh_eles_mpo(mpo, L, R, bxs, bys):
    nx, ny = len(bxs), len(bys)
    fs = np.zeros((nx, ny), dtype=mpo[0].dtype)
    for i in prange(nx):  # Parallelize this loop
        for j in prange(ny):
            bstr = bxs[i] + bys[j]  # Combine binary numbers directly
            fs[i, j] = get_ele_mpo(mpo, L, R, bstr)
    return fs
