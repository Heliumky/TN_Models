import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../CytnxTools')))
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
import twobody as twut

if __name__ == '__main__':
    N = 6
    #cutoff = 0.1
    shift = -20
    rescale = -2*shift/(2**N-1)
    xmax = -shift
    cutoff = rescale
    dx = rescale 
    dtype = float
    Rad = 1.4
    parity = 0
    print('xmax =',xmax)
    print('xshift =',shift)
    print('dx =',dx)
    tci_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../CytnxTools/tci.py'))
    # One-body H
    Hk1, Lk1, Rk1 = hamilt.H_kinetic(N,dx)
    Hk1, Lk1, Rk1 = hamilt.get_H_3D (Hk1, Lk1, Rk1)
    # Potential energy
    factor = 1
    
    os.system(f'python3 {tci_path} {3*N} {rescale} {shift} {cutoff} {factor} {-Rad} --3D_one_over_r')
    
    V1 = tci.load_mps(f'fit{3*N}_{-Rad}.mps.npy')
    HV1, LV1, RV1 = npmps.mps_func_to_mpo(V1)
    
    
    os.system(f'python3 {tci_path} {3*N} {rescale} {shift} {cutoff} {factor} {Rad} --3D_one_over_r')
    V11 = tci.load_mps(f'fit{3*N}_{Rad}.mps.npy')
    HV11, LV11, RV11 = npmps.mps_func_to_mpo(V11)

    V2 = tci.load_mps(f'fit{3*N}_{Rad}.mps.npy')
    HV2, LV2, RV2 = npmps.mps_func_to_mpo(V2)

    V22 = tci.load_mps(f'fit{3*N}_{-Rad}.mps.npy')
    HV22, LV22, RV22 = npmps.mps_func_to_mpo(V22)


    H1, L1, R1 = npmps.sum_2MPO (Hk1, Lk1, Rk1, HV1, LV1, RV1)
    H1, L1, R1 = npmps.sum_2MPO (H1, L1, R1, HV11, LV11, RV11)
    H2, L2, R2 = npmps.sum_2MPO (Hk1, Lk1, Rk1, HV2, LV2, RV2)
    H2, L2, R2 = npmps.sum_2MPO (H2, L2, R2, HV22, LV22, RV22)
    H, L, R = twut.H_two_particles (H1, L1, R1, H2, L2, R2)

    # Interaction
    #os.system(f'python3 {tci_path} {6*N} {rescale} {shift} {cutoff} {factor} {0} --3D_interact_Hmol')
    inter_MPS  = tci.load_mps(f'fit{6*N}_{0.0}.mps.npy')
    Hint, Lint, Rint = npmps.mps_func_to_mpo (inter_MPS)
    Hint = twut.H_merge_two_particle (Hint)
    print("len of Hint:", len(Hint))

    compress_cf = 1e-8
    H, L, R = npmps.sum_2MPO (H, L, R, Hint, Lint, Rint)

    H, L, R = npmps.compress_MPO (H, L, R, cutoff=compress_cf)

    print("len of H:", len(H))

    # Rotate H to the parity basis
    U = twut.get_swap_U()
    H = ut.applyLocalRot_mpo (H, U)

    # Set quantum numbers
    H, L, R = twut.set_mpo_quantum_number (H, L, R, parity)

    # Create MPS
    psi = twut.make_product_mps (3*N, parity)


    # Define the bond dimensions for the sweeps
    #maxdims = [2]*10 + [4]*10
    maxdims = [2]*10 + [4]*40 + [8]*50 + [16]*20 + [32]*10 

    c0 = mpsut.inner (psi, psi)

    # Run dmrg
    psi, ens, terrs0 = dmrg.dmrg (2, psi, H, L, R, maxdims, compress_cf, maxIter=4)
    np.savetxt(f'3d_terr0_N={N}_par={parity}.dat',(maxdims,terrs0,ens))
    # ------- Compute the single-particle correlation function -----------
    psi = ut.mps_to_nparray (psi)
    # --------------- save ---------------------

    data = {
        'Rad': Rad,
        'parity':parity,
        'N': N,
        'shift': shift,
        'rescale': rescale, 
        'xmax': xmax,  
        'cutoff': rescale, 
        'Energy': ens[-1],
        'psi':  psi,
    }
    with open(f'3d_dmrg_N={N}_par={parity}.pkl', 'wb') as f:
        pickle.dump(data, f)



