import dmrg as dmrg
import numpy as np
import qtt_utility as ut
import os
import MPS_utility as mpsut
import hamilt
from test import load_mps
import npmps
import pickle
import plot_utility as ptut
import matplotlib.pyplot  as plt

if __name__ == '__main__':
    N = 3
    #cutoff = 0.1
    shift = - 4
    rescale = -2*shift/(2**N-1)
    xmax = -shift
    cutoff = rescale
    dx = rescale 
    print('xmax =',xmax)
    print('xshift =',shift)
    print('dx =',dx)



    # Generate the MPS for the potential


    # Kinetic energy
    Hk1, Lk1, Rk1 = hamilt.H_kinetic(N, dx)
    Hk, Lk, Rk = hamilt.get_H_2D (Hk1, Lk1, Rk1)
    assert len(Hk) == 2*N

    # Potential energy
    factor = 1
    os.system('python3 tci.py '+str(2*N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --2D_one_over_r')
    V_MPS = load_mps(f'fit{2*N}.mps.npy')
    #V_MPS = tci.tci_one_over_r_2D (2*N, rescale, cutoff, factor, shift)
    HV, LV, RV = npmps.mps_func_to_mpo(V_MPS)

    '''bxs = list(ptut.BinaryNumbers(N))
    bys = list(ptut.BinaryNumbers(N))
    xs = ptut.bin_to_dec_list (bxs)
    ys = ptut.bin_to_dec_list (bys)
    X, Y = np.meshgrid (xs, ys)
    ZV = ptut.get_2D_mesh_eles_mps (V_MPS, bxs, bys)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, ZV, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()'''

    H, L, R = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)
    assert len(H) == 2*N

    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)

    H = npmps.absort_LR (H, L, R)

    Hmtx = npmps.MPO_to_matrix (H)
    print(Hmtx.shape)
    from scipy.sparse.linalg import eigsh, eigs
    print(Hmtx)
    eigenvalues, eigenvectors = np.linalg.eigh(Hmtx)
    print(eigenvalues[0])

