import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
import dmrg as dmrg
import qtt_utility as ut
import MPS_utility as mpsut
import hamilt
import tci
import npmps
import pickle
import plot_utility as ptut
import matplotlib.pyplot  as plt
import numpy as np
import time



if __name__ == '__main__':
    N = 8
    #cutoff = 0.1
    shift = - 10
    rescale = -2*shift/(2**N-1)
    xmax = -shift
    cutoff = rescale
    dx = rescale 
    print('xmax =',xmax)
    print('xshift =',shift)
    print('dx =',dx)
    dtype = float
    # Generate the MPS for the potential


    # Kinetic energy
    Hk1, Lk1, Rk1 = hamilt.H_kinetic(N, dx)
    Hk, Lk, Rk = hamilt.get_H_3D (Hk1, Lk1, Rk1)
    assert len(Hk) == 3*N
    start = time.time()
    # Potential energy
    factor = 1
    #os.system('python3 tci.py '+str(3*N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --3D_one_over_r')
    #V_MPS = load_mps('fit.mps.npy')
    V_MPS = tci.load_mps(f'fit{3*N}.mps.npy')
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
    assert len(H) == 3*N

    H, L, R = npmps.compress_MPO (H, L, R, cutoff=1e-12)

    # Create MPS

    # -------------- Energy arr -------
    energy_arr= []
    # -------------- psi arr ----------
    phi_arr = []
    # -------------- DMRG -------------
    H, L, R = mpsut.npmpo_to_uniten (H, L, R)

    # Define the bond dimensions for the sweeps
    #maxdims = [2]*10 + [4]*10 + [8]*40 + [16]*40 + [32]*40
    cutoff = 1e-12

    # Run dmrg
    maxdims = [2]*10 + [4]*40 + [8]*80 + [16]*80 + [32]*80
    #maxdims = [2]*10
    psi0 = npmps.random_MPS (3*N, 2, 2, dtype=dtype)
    psi0 = mpsut.npmps_to_uniten (psi0)
    psi0, ens0, terrs0 = dmrg.dmrg (2, psi0, H, L, R, maxdims, cutoff)
    energy_arr.append(ens0[-1])
    phi0 = mpsut.to_npMPS (psi0)
    phi_arr.append(phi0)
    np.savetxt(f'3d_terr0_N={N}.dat',(maxdims,terrs0,ens0))

    #maxdims = maxdims + [64]*80
    maxdims = [2]*10 + [4]*40 + [8]*120 + [16]*180 + [32]*100
    #maxdims = [2]*10
    psi1 = npmps.random_MPS (3*N, 2, 2, dtype=dtype)
    psi1 = mpsut.npmps_to_uniten (psi1)
    psi1,ens1,terrs1 = dmrg.dmrg (2, psi1, H, L, R, maxdims, cutoff, ortho_mpss=[psi0], weights=[10])
    energy_arr.append(ens1[-1])
    phi1 = mpsut.to_npMPS (psi1)
    phi_arr.append(phi1)
    np.savetxt(f'3d_terr1_N={N}.dat',(maxdims,terrs1,ens1))


    maxdims = maxdims + [64]*256
    #maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    psi2 = npmps.random_MPS (3*N, 2, 2, dtype=dtype)
    psi2 = mpsut.npmps_to_uniten (psi2)
    psi2,ens2,terrs2 = dmrg.dmrg (2, psi2, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1], weights=[20,20])
    energy_arr.append(ens2[-1])
    phi2 = mpsut.to_npMPS (psi2)
    phi_arr.append(phi2)
    np.savetxt(f'3d_terr2_N={N}.dat',(maxdims,terrs2,ens2))

    #maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    psi3 = npmps.random_MPS (3*N, 2, 2, dtype=dtype)
    psi3 = mpsut.npmps_to_uniten (psi3)
    psi3,ens3,terrs3 = dmrg.dmrg (2, psi3, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1,psi2], weights=[20,20,20])
    energy_arr.append(ens3[-1])
    phi3 = mpsut.to_npMPS (psi3)
    phi_arr.append(phi3)
    np.savetxt(f'3d_terr3_N={N}.dat',(maxdims,terrs3,ens3))

    #maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    psi4 = npmps.random_MPS (3*N, 2, 2, dtype=dtype)
    psi4 = mpsut.npmps_to_uniten (psi4)
    psi4,ens4,terrs4 = dmrg.dmrg (2, psi4, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1,psi2,psi3], weights=[20,20,20,20])
    energy_arr.append(ens4[-1])
    phi4 = mpsut.to_npMPS (psi4)
    phi_arr.append(phi4)
    np.savetxt(f'3d_terr4_N={N}.dat',(maxdims,terrs4,ens4))
  
    #maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    #psi5 = npmps.random_MPS (3*N, 2, 2, dtype=dtype)
    #psi5 = mpsut.npmps_to_uniten (psi5)
    #psi5,ens5,terrs5 = dmrg.dmrg (2, psi5, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1,psi2,psi3,psi4], weights=[20,20,20,20,20])
    #energy_arr.append(ens5[-1])
    #phi5 = mpsut.to_npMPS (psi5)
    #phi_arr.append(phi5)
    #np.savetxt('3d_terr5_N={N}.dat',(maxdims,terrs5,ens5))
    end = time.time()
    print(f"TCI + DMRG time: {end - start:.5f} seconds")
    print(ens0[-1], ens1[-1], ens2[-1], ens3[-1], ens4[-1])
    #print(ens0[-1], ens1[-1])

    data = {
        'N': N,
        'shift': shift,
        'rescale': rescale, 
        'xmax': xmax,  
        'cutoff': rescale, 
        'dx': dx, 
        'energy_arr': energy_arr,
        'phi_arr': phi_arr,
        'V_MPS': V_MPS,
    }
    with open(f'3d_dmrg_results_N={N}.pkl', 'wb') as f:
        pickle.dump(data, f)
