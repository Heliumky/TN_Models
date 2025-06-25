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

if __name__ == '__main__':
    N = 3
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
    print(len(psi))

    # Define the bond dimensions for the sweeps
    #maxdims = [2]*10 + [4]*50 + [8]*50 + [16]*50 + [32]*40
    maxdims = [2]*1
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
    corr, Lcorr, Rcorr = npmps.compress_MPO (corr, Lcorr, Rcorr, cutoff=1e-12)
    corr, Lcorr, Rcorr = npmps.change_dtype (corr, Lcorr, Rcorr, dtype=complex)

    #maxdims = [2]*10 + [4]*50 + [8]*50 + [16]*50
    #maxdims = [2]*1
    corr, Lcorr, Rcorr = ut.mpo_to_uniten (corr, Lcorr, Rcorr)
    phi1 = npmps.random_MPS(N, 2, 15, 2,complex)
    phi1 = ut.mps_to_uniten (phi1)
    phi1, occ1,terrs1 = dmrg.dmrg (phi1, corr, Lcorr, Rcorr, maxdims, cutoff)

    #maxdims = [2]*10 + [4]*50 + [8]*50 + [16]*50
    maxdims = [2]*1
    phi2 = npmps.random_MPS(N, 2, 15, 2,complex)
    phi2 = ut.mps_to_uniten (phi2)
    phi2, occ2, terrs2 = dmrg.dmrg (phi2, corr, Lcorr, Rcorr, maxdims, cutoff, ortho_mpss=[phi1], weights=[1])

    overlap = mpsut.inner (phi1, phi2)

    print('E =',ens[-1])
    print('occ =',occ1[-1], occ2[-1])
    print('inner product of the orbitals =', overlap)

    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))
    xs = ptut.bin_to_dec_list (bxs)
    xs = np.asarray(xs)
    xs = xs*rescale + shift

    # The potential
    Vx = [ptut.get_ele_mpo (HV, LV, RV, bx) for bx in bxs]

    # First particle
    npphi1 = mpsut.to_npMPS (phi1)
    ys1 = [ptut.get_ele_mps (npphi1, bx) for bx in bxs]

    # Second particle
    npphi2 = mpsut.to_npMPS (phi2)
    ys2 = [ptut.get_ele_mps (npphi2, bx) for bx in bxs]

    fig, ax = plt.subplots()
    #ax.plot (xs, Vx)
    ax.plot (xs, np.abs(ys1/np.sqrt(dx))**2, marker='.')
    ax.plot (xs, np.abs(ys2/np.sqrt(dx))**2, marker='.')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|\phi(x)|^2$')
    #ps.set(ax)
    fig.savefig('phi.pdf')
    plt.show()

