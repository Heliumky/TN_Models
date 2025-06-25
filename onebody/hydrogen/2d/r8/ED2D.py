import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CytnxTools')))
import numpy as np
import os
import hamilt
import tci
import npmps
import plot_utility_jit as ptut
import matplotlib.pyplot as plt
import pickle

def eigenvec_to_mps(eigenvec, xsite, ysite, d=2, max_bond_dim=None):
   
    L = xsite + ysite
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
    N = 6
    #cutoff = 0.1
    shift = -8
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
    #os.system('python3 tci.py '+str(2*N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --2D_one_over_r')
    V_MPS = tci.load_mps(f'fit{2*N}.mps.npy')
    #V_MPS = tci.tci_one_over_r_2D (2*N, rescale, cutoff, factor, shift)
    HV, LV, RV = npmps.mps_func_to_mpo(V_MPS)

    H, L, R = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)
    assert len(H) == 2*N

    H, L, R = npmps.compress_MPO (H, L, R, cutoff=1e-12)

    H = npmps.absort_LR (H, L, R)
    print([H[i].shape for i in range(len(H))])
    Hmtx = npmps.MPO_to_matrix (H)
    print(Hmtx.shape)

    import scipy as sp
    #print(Hmtx)
    
    eigenvalues, eigenvectors = sp.linalg.eigh(Hmtx,  subset_by_index=[0, 3])   # for small size
    #eigenvalues, eigenvectors = np.linalg.eigh(Hmtx)   # for large size
    #eigenvalues = eigenvalues[0:4]
   
    print(eigenvalues[0])
    print(eigenvalues[1])
    print(eigenvalues[2])
    print(eigenvalues[3])

    mps0 = eigenvec_to_mps(eigenvectors[:,0], N, N, 2)
    mps1 = eigenvec_to_mps(eigenvectors[:,1], N, N, 2)
    mps2 = eigenvec_to_mps(eigenvectors[:,2], N, N, 2)
    mps3 = eigenvec_to_mps(eigenvectors[:,3], N, N, 2)
    mps_arr = [mps0, mps1, mps2, mps3]
    
    data = {
        'N': N,
        'shift': shift,
        'rescale': rescale, 
        'xmax': xmax,  
        'cutoff': rescale, 
        'dx': dx, 
        'eigvals': eigenvalues,
        'mps_arr': mps_arr,
        'V_MPS': V_MPS,
    }
    with open(f'2d_ED_results_N={N}.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    #print([mps1[i].shape for i in range(len(mps1))])
    #print(np.sum(np.abs(eigenvectors[0])**2))
    #print(npmps.inner_MPS(mps1,mps1))
    #print(max(np.abs(eigenvectors[0]*dx)**2))
    '''
    bxs = list(ptut.BinaryNumbers(N))
    bys = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list (bxs,rescale=rescale, shift=shift)
    ys = ptut.bin_to_dec_list (bys,rescale=rescale, shift=shift)
    X, Y = np.meshgrid (xs, ys)

    ZV = ptut.get_2D_mesh_eles_mps (V_MPS, bxs, bys)
    Z0 = ptut.get_2D_mesh_eles_mps (mps0, bxs, bys)
    Z1 = ptut.get_2D_mesh_eles_mps (mps1, bxs, bys)
    Z2 = ptut.get_2D_mesh_eles_mps (mps2, bxs, bys)
    Z3 = ptut.get_2D_mesh_eles_mps (mps3, bxs, bys)

    plt.figure(figsize=(16, 8))

    wavefunctions = [Z0, Z1, Z2, Z3]
    titles = ['1s', '2p', '2p', '2s']
    cmaps = ['viridis', 'plasma', 'plasma', 'plasma']

    for i, (Z, title, cmap) in enumerate(zip(wavefunctions, titles, cmaps)):
        # 实部
        plt.subplot(2, 4, i + 1)
        plt.imshow(Z.real/dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
        plt.colorbar(label=f'$Re(\psi)$')
        plt.title(f'{title} Real Part\nEnergy = {eigenvalues[i]:.8f}')
        plt.xlabel('x')
        plt.ylabel('y')

        # 虚部
        #plt.subplot(3, 4, i + 5)
        #plt.imshow(Z.imag/dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
        #plt.colorbar(label=f'$Im(\psi)$')
        #plt.title(f'{title} Imag Part\nEnergy = {eigenvalues[i]:.8f}')
        #plt.xlabel('x')
        #plt.ylabel('y')

        #plt.subplot(3, 4, i + 9)
        plt.subplot(3, 4, i + 5)
        plt.imshow(np.abs(Z/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
        plt.colorbar(label=f'$|\psi|^2$')
        plt.title(f'{title} Density\nEnergy = {eigenvalues[i]:.8f}')
        plt.xlabel('x')
        plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f"2d_extdiag_{N}.pdf", format='pdf')
    plt.show()
    '''
