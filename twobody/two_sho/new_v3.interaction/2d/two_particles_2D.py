import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../CytnxTools')))
import dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import qtt_utility as ut
import copy, sys, os
import MPS_utility as mpsut
import twobody as twut
import plot_utility as ptut
from matplotlib import cm
import hamilt
import tci
import npmps
import pickle

if __name__ == '__main__':
    N = 8
    #cutoff = 0.1
    shift = - 7
    rescale = -2*shift/(2**N-1)
    xmax = -shift
    cutoff = rescale
    dx = rescale 
    dtype = float
    parity = 0
    print('xmax =',xmax)
    print('xshift =',shift)
    print('dx =',dx)
    tci_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../CytnxTools/tci.py'))


    # Generate the MPS for the potential


    # Kinetic energy
    #Hk1, Lk1, Rk1 = hamilt.H_kinetic(N, dx)
    Hk, Lk, Rk = hamilt.H_kinetic(N, dx)
    HV, LV, RV = hamilt.H_trap(N, dx, shift)
    #Hk, Lk, Rk = hamilt.get_H_2D (Hk1, Lk1, Rk1)

    # Potential energy
    factor = 1.

    #H1, L1, R1 = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)
    print(len(Hk),len(HV))
    #H1, L1, R1 = npmps.sum_2MPO (Hk, 0.5*Lk, Rk, HV, 2*LV, RV)
    H1, L1, R1 = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)
    H1, L1, R1 = hamilt.get_H_2D (H1, L1, R1)
    #H2, L2, R2 = npmps.sum_2MPO (Hk, 2*Lk, Rk, HV, 1.5*LV, RV)
    H2, L2, R2 = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)
    H2, L2, R2 = hamilt.get_H_2D (H2, L2, R2)

    assert len(H1) == 2*N

    H, L, R = twut.H_two_particles (H1, L1, R1, H2, L2, R2)

    # Interaction
    
    os.system(f'python3 {tci_path} {4*N} {rescale} {shift} {cutoff} {factor} {0} --2D_interact_SHO')
    inter_MPS  = tci.load_mps(f'fit{4*N}_{0.0}.mps.npy')
    Hint, Lint, Rint = npmps.mps_func_to_mpo (inter_MPS)
    Hint = twut.H_merge_two_particle (Hint)
    print("len of Hint:", len(Hint))

    H, L, R = npmps.sum_2MPO (H, L, R, Hint, Lint, Rint)

    H, L, R = npmps.compress_MPO (H, L, R, cutoff=1e-12)

    print("len of H:", len(H))
 
    # Rotate H to the parity basis
    U = twut.get_swap_U()
    H = ut.applyLocalRot_mpo (H, U)

    # Set quantum numbers
    H, L, R = twut.set_mpo_quantum_number (H, L, R, parity)

    # Create MPS
    psi = twut.make_product_mps (2*N, parity)


    # Define the bond dimensions for the sweeps
    #maxdims = [2]*10 + [4]*10 + [8]*40 + [16]*40 + [32]*40 + [64]*40
    maxdims = [2]*10 + [4]*40 + [8]*50 + [16]*50 + [32]*30 
    cutoff = 1e-12

    c0 = mpsut.inner (psi, psi)

    # Run dmrg
    psi, ens, terrs0 = dmrg.dmrg (2,psi, H, L, R, maxdims, cutoff, maxIter=4)
    np.savetxt(f'2d_terr0_N={N}_par={parity}.dat',(maxdims,terrs0,ens))
    # ------- Compute the single-particle correlation function -----------
    psi = ut.mps_to_nparray (psi)
    psi = ut.applyLocal_mps (psi, U)

    corr = twut.corr_matrix (psi)
    Lcorr = np.array([1.])
    Rcorr = np.array([1.])

    # Target the largest occupations
    corr[0] *= -1.
    corr, Lcorr, Rcorr = npmps.compress_MPO (corr, Lcorr, Rcorr, cutoff=1e-12)

    maxdims = [2]*10  + [4]*10 + [8]*40 + [16]*40 + [32]*40 

    corr, Lcorr, Rcorr = ut.mpo_to_uniten (corr, Lcorr, Rcorr)
    phi1 = npmps.random_MPS (2*N, 2, 15, dtype=dtype)
    phi1 = ut.mps_to_uniten (phi1)

    phi1, occ1, terrs1 = dmrg.dmrg (2, phi1, corr, Lcorr, Rcorr, maxdims, cutoff)
    np.savetxt(f'2d_terr1_N={N}_par={parity}.dat',(maxdims,terrs1,occ1))
    phi2 = npmps.random_MPS (2*N, 2, 15, dtype=dtype)
    phi2 = ut.mps_to_uniten (phi2)
    phi2, occ2, terrs2 = dmrg.dmrg (2, phi2, corr, Lcorr, Rcorr, maxdims, cutoff, ortho_mpss=[phi1], weights=[20])
    np.savetxt(f'2d_terr2_N={N}_par={parity}.dat',(maxdims,terrs2,occ2))
    # --------------- Plot ---------------------


    # First particle
    npphi1 = mpsut.to_npMPS (phi1)


    # Second particle
    npphi2 = mpsut.to_npMPS (phi2)

    data = {
        'N': N,
        'shift': shift,
        'rescale': rescale, 
        'xmax': xmax,  
        'cutoff': rescale, 
        'dx': dx, 
        'occ1': occ1[-1],
        'occ2': occ2[-1],
        'Energy': ens[-1],
        'phi1': npphi1,
        'phi2': npphi2,
        'psi':  psi,
    }
    with open(f'2d_dmrg_N={N}_par={parity}.pkl', 'wb') as f:
        pickle.dump(data, f)


    bxs = list(ptut.BinaryNumbers(N))
    bys = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list (bxs,rescale=rescale, shift=shift)
    ys = ptut.bin_to_dec_list (bys,rescale=rescale, shift=shift)
    X, Y = np.meshgrid (xs, ys)

    #ZV = ptut.get_2D_mesh_eles_mps (V_MPS, bxs, bys)
    Z0 = ptut.get_2D_mesh_eles_mps (npphi1 , bxs, bys)
    Z1 = ptut.get_2D_mesh_eles_mps (npphi2, bxs, bys)
    #Z2 = ptut.get_2D_mesh_eles_mps (phi2, bxs, bys)
    #Z3 = ptut.get_2D_mesh_eles_mps (phi3, bxs, bys)

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface (X, Y, Z0, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface (X, Y, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface (X, Y, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface (X, Y, Z3, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Ground state density
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(Z0/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title(f'phi1 Density (occ = {occ1[-1]:.8f})')
    plt.xlabel('x')
    plt.ylabel('y')

# Excited state density
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(Z1/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='plasma')
    plt.colorbar(label='Intensity')
    plt.title(f'phi2 Density (occ = {occ2[-1]:.8f})')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig(f"2d_dmrg_density_functions{N}_par={parity}.pdf", format='pdf')
    plt.show()


