import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
import dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import qtt_utility as ut
import copy, sys, os
import MPS_utility as mpsut
import twobody as twut
import SHO as sho
import plot_utility as ptut
from matplotlib import cm
import hamilt
import tci
import npmps
import cytnx
#import plotsetting as ps

def eigenvec_to_mps(eigenvec, xsite, d=2, max_bond_dim=None):
   
    L = xsite
    assert len(eigenvec) == d**L
    
    tensor_shape = [d] * L
    remaining = eigenvec.reshape(tensor_shape)
        
    bond_dim = 1  
    
    mps = []
    for i in range(L-1):

        remaining = remaining.reshape(bond_dim * d, -1)
        

        U, S, Vh = np.linalg.svd(remaining, full_matrices=False)
        norm = np.sqrt(np.sum(S**2))
        S = S / norm

        if max_bond_dim is not None and len(S) > max_bond_dim:
            U = U[:, :max_bond_dim]
            S = S[:max_bond_dim]
            Vh = Vh[:max_bond_dim, :]
        
        new_bond_dim = U.shape[1]

        mps_tensor = U.reshape(bond_dim, d, new_bond_dim)
        mps.append(mps_tensor)

        remaining = np.dot(np.diag(S), Vh)
        bond_dim = new_bond_dim

    mps.append(remaining.reshape(bond_dim, d, 1))
    return mps

if __name__ == '__main__':
    N = 9
    #cutoff = 0.1
    shift = - 10
    rescale = -2*shift/(2**N-1)
    xmax = -shift
    cutoff = rescale
    dx = rescale 
    print('xmax =',xmax)
    print('xshift =',shift)
    print('dx =',dx)



    # Generate the MPS for the potential


    # Kinetic energy
    Hk, Lk, Rk = hamilt.H_kinetic(N, dx)
    HV, LV, RV = hamilt.H_trap(N, dx, shift)
    # Potential energy
    #factor = 1.
    #os.system('python3 tci.py '+str(N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --1D_one_over_r')
    # Load the potential MPS
    #V_MPS = load_mps('fit.mps.npy')
    #HV, LV, RV = npmps.mps_func_to_mpo(V_MPS)

    print(len(Hk),len(HV))
    H1, L1, R1 = npmps.sum_2MPO (Hk, 0.5*Lk, Rk, HV, 2*LV, RV)
    H2, L2, R2 = npmps.sum_2MPO (Hk, 2*Lk, Rk, HV, 1.5*LV, RV)

    # Create a two-particle Hamiltonian
    H, L, R = twut.H_two_particles (H1, L1, R1, H2, L2, R2)

    H, L, R = npmps.compress_MPO (H, L, R, cutoff=1e-12)
    H, L, R = npmps.change_dtype (H, L, R, dtype=complex)
    # Rotate H to the parity basis
    U = twut.get_swap_U()
    H = ut.applyLocalRot_mpo (H, U)

    # Set quantum numbers
    H, L, R = twut.set_mpo_quantum_number (H, L, R)
    
    # Create MPS
    psi = twut.make_product_mps (N,1)  # 1 complex, 3 double
    #psi = twut.make_product_mps (N, cytnx.Type.Double)
    print([psi[i].dtype() for i in range(len(psi))])

    # Define the bond dimensions for the sweeps
    maxdims = [2]*10 + [4]*50 + [8]*50 + [16]*50 + [32]*40
    #maxdims = [2]*10
    cutoff = 1e-12

    c0 = mpsut.inner (psi, psi)
    print("c0 =", c0)
    # Run dmrg
    psi, ens, terrs = dmrg.dmrg (psi, H, L, R, maxdims, cutoff, maxIter=4)

    # ------- Compute the single-particle correlation function -----------
    psi = ut.mps_to_nparray (psi)
    psi = ut.applyLocal_mps (psi, U)

    corr = twut.corr_matrix (psi)
    Lcorr = np.array([1.])
    Rcorr = np.array([1.])

    # Target the largest occupations
    corr[0] *= -1.
    corrmtx = npmps.MPO_to_matrix (corr)
    #occ, phi = sp.linalg.eigh(corrmtx,  subset_by_index=[0, 1])
    occ, phi = np.linalg.eigh(corrmtx)

    phi1 = eigenvec_to_mps(phi[:,0], N, d=2, max_bond_dim=None)
    phi2 = eigenvec_to_mps(phi[:,1], N, d=2, max_bond_dim=None)

    bxs = list(ptut.BinaryNumbers(N))
    xs = ptut.bin_to_dec_list (bxs)
    xs = np.asarray(xs)
    xs = xs*rescale + shift

    Z0 = np.array([ptut.get_ele_mps (phi1, bx) for bx in bxs])
    Z1 = np.array([ptut.get_ele_mps (phi2, bx) for bx in bxs])
    wavefunctions = [Z0, Z1]
    titles = ['phi1', 'phi2']
    cmaps = ['viridis', 'plasma']
    overlap = npmps.inner_MPS(phi1, phi2)
    print('E =',ens[-1])
    print('occ =',occ[0], occ[1])
    print('inner product of the orbitals =', overlap)

    # --------------- Plot ---------------------
    plt.figure(figsize=(8, 12))
    for i, (Z, title, cmap) in enumerate(zip(wavefunctions, titles, cmaps)):
        # 实部
        plt.subplot(3, 2, i + 1)
        plt.plot (xs, Z.real/np.sqrt(dx), marker='.')
        plt.title(f'{title} Real Part\nocc = {occ[i]:.8f}')
        plt.xlabel('x')
        plt.ylabel(f'Re$\phi$')

        # 虚部
        plt.subplot(3, 2, i + 3)
        plt.plot (xs, Z.imag/np.sqrt(dx), marker='.')
        plt.title(f'{title} Imag Part\nocc = {occ[i]:.8f}')
        plt.xlabel('x')
        plt.ylabel(f'Imag$\phi$')

        plt.subplot(3, 2, i + 5)
        plt.plot (xs, np.abs(Z/np.sqrt(dx))**2, marker='.')
        plt.title(f'{title} Density\nocc = {occ[i]:.8f}')
        plt.xlabel('x')
        plt.ylabel(f'$|\phi|^2$')
    plt.tight_layout()
    plt.savefig(f"1d_extdiag_{N}.pdf", format='pdf')
    plt.show()


