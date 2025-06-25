import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
import dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import qtt_utility as ut
import copy, sys, os
import qn_utility as qn
import MPS_utility as mpsut
import twobody as twut
import SHO as sho
import plot_utility as ptut
from matplotlib import cm
import hamilt
import tci
import npmps
import pickle

if __name__ == '__main__':
    N = 8
    #cutoff = 0.1
    N_2D = 2*N
    shift = - 7
    rescale = -2*shift/(2**N-1)
    xmax = -shift
    cutoff = rescale
    dx = rescale 
    print('xmax =',xmax)
    print('xshift =',shift)
    print('dx =',dx)



    # Generate the MPS for the potential


    # Kinetic energy
    #Hk1, Lk1, Rk1 = hamilt.H_kinetic(N, dx)
    Hk, Lk, Rk = hamilt.H_kinetic(N, dx)
    HV, LV, RV = hamilt.H_trap(N, dx, shift)
    #Hk, Lk, Rk = hamilt.get_H_2D (Hk1, Lk1, Rk1)

    # Potential energy
    factor = 1.
    #os.system('python3 tci.py '+str(2*N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --2D_one_over_r')
    #V_MPS = load_mps('fit.mps.npy')
    #HV, LV, RV = npmps.mps_func_to_mpo(V_MPS)
    #assert len(V_MPS) == 2*N

    #H1, L1, R1 = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)
    print(len(Hk),len(HV))
    #H1, L1, R1 = npmps.sum_2MPO (Hk, 0.5*Lk, Rk, HV, 2*LV, RV)
    H1, L1, R1 = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)
    H1, L1, R1 = hamilt.get_H_2D (H1, L1, R1)
    #H2, L2, R2 = npmps.sum_2MPO (Hk, 2*Lk, Rk, HV, 1.5*LV, RV)
    H2, L2, R2 = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)
    H2, L2, R2 = hamilt.get_H_2D (H2, L2, R2)

    assert len(H1) == 2*N

    # Create a two-particle Hamiltonian
    H, L, R = twut.H_two_particles (H1, L1, R1, H2, L2, R2)
    assert len(H) == N_2D


    H, L, R = npmps.compress_MPO (H, L, R, cutoff=1e-12)
    H, L, R = npmps.change_dtype (H, L, R, dtype=complex)
    # Rotate H to the parity basis
    U = twut.get_swap_U()
    H = ut.applyLocalRot_mpo (H, U)

    # Set quantum numbers
    H, L, R = twut.set_mpo_quantum_number (H, L, R)

    # Create MPS
    psi = twut.make_product_mps (N_2D)


    # Define the bond dimensions for the sweeps
    maxdims = [2]*10 + [4]*20 + [8]*60 + [16]*80
    #maxdims = [2]*10 
    cutoff = 1e-12

    c0 = mpsut.inner (psi, psi)

    # Run dmrg
    psi, ens, terrs = dmrg.dmrg (psi, H, L, R, maxdims, cutoff, maxIter=4)
    np.savetxt(f'2d_terrs_N={N}.dat',(maxdims, terrs,ens))
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
    maxdims = [2]*10 + [4]*40 + [8]*60 + [16]*80
    #maxdims = [2]*10 
    corr, Lcorr, Rcorr = ut.mpo_to_uniten (corr, Lcorr, Rcorr)
 
    phi1 = ut.generate_random_MPS_nparray (N_2D, d=2, D=2)
    phi1 = ut.mps_to_uniten (phi1)

    phi1, occ1, terrs1 = dmrg.dmrg (phi1, corr, Lcorr, Rcorr, maxdims, cutoff)
    np.savetxt(f'2d_terr1_N={N}.dat',(maxdims, terrs1, occ1))
    phi2 = ut.generate_random_MPS_nparray (N_2D, d=2, D=2)
    phi2 = ut.mps_to_uniten (phi2)
    phi2, occ2, terrs2 = dmrg.dmrg (phi2, corr, Lcorr, Rcorr, maxdims, cutoff, ortho_mpss=[phi1], weights=[20])
    np.savetxt(f'2d_terr2_N={N}.dat',(maxdims, terrs2, occ2))
    overlap = mpsut.inner (phi1, phi2)

    print('E =',ens[-1])
    print('occ =',occ1[-1], occ2[-1])
    print('inner product of the orbitals =', overlap)

    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))
    bys = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list (bxs)
    xs = np.asarray(xs)
    xs = xs*rescale + shift
    ys = ptut.bin_to_dec_list (bys)
    ys = np.asarray(ys)
    ys = ys*rescale + shift
    X, Y = np.meshgrid (xs, ys)

    # First particle
    npphi1 = mpsut.to_npMPS (phi1)
    Z1 = ptut.get_2D_mesh_eles_mps (npphi1, bxs, bys)

    # Second particle
    npphi2 = mpsut.to_npMPS (phi2)
    Z2 = ptut.get_2D_mesh_eles_mps (npphi2, bxs, bys)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.savefig('phi1.pdf')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.savefig('phi2.pdf')
    plt.show()

    data = {
        'N': N,
        'shift': shift,
        'rescale': rescale, 
        'xmax': xmax,  
        'cutoff': rescale, 
        'dx': dx, 
        'energy_1s': ens[-1],
        'occ1': occ1[-1],
        'occ2': occ2[-1],
        'Energy': ens,
        'phi1': npphi1,
        'phi2': npphi2,
        'psi':  psi,
    }
    with open(f'2d_dmrg_N={N}.pkl', 'wb') as f:
        pickle.dump(data, f)
