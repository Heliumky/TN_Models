import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
import matplotlib.pyplot as plt
import plot_utility_jit as ptut
import hamilt
import tci
import npmps
import numpy as np


def eigenvec_to_mps(eigenvec, xsite, ysite, zsite, d=2, max_bond_dim=None):
   
    L = xsite + ysite + zsite
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
    N = 5
    #cutoff = 0.1
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
    Hk1, Lk1, Rk1 = hamilt.H_kinetic(N, dx)
    Hk, Lk, Rk = hamilt.get_H_3D (Hk1, Lk1, Rk1)
    assert len(Hk) == 3*N

    # Potential energy
    factor = 1
    os.system('python3 tci.py '+str(3*N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --3D_one_over_r')
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

    H = npmps.absort_LR (H, L, R)

    Hmtx = npmps.MPO_to_matrix (H)
    print(Hmtx.shape)
    import scipy as sp
    #print(Hmtx)
    eigenvalues, eigenvectors = sp.linalg.eigh(Hmtx,  subset_by_index=[0, 4])
    print(eigenvalues[0])
    print(eigenvalues[1])
    print(eigenvalues[2])
    print(eigenvalues[3])
    print(eigenvalues[4])
    
    # 假设已经有 eigenvalues 和 eigenvectors
    eigenpairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(5)]  # 取前4个态
    state_labels = ['1s', '2s', '2p', '2p', '2p']  # 自己定义每个态的名字

    bxs = list(ptut.BinaryNumbers(N))
    bys = list(ptut.BinaryNumbers(N))
    bzs = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list(bxs, rescale=rescale, shift=shift)
    ys = ptut.bin_to_dec_list(bys, rescale=rescale, shift=shift)
    zs = ptut.bin_to_dec_list(bzs, rescale=rescale, shift=shift)
    X, Y, Z = np.meshgrid(xs, ys, zs)

    for idx, (energy, eigvec) in enumerate(eigenpairs):
        phi = eigenvec_to_mps(eigvec, N, N, N, 2)
        Zphi = ptut.get_3D_mesh_eles_mps(phi, bxs, bys, bzs)

        # Normalize density
        norm_factor = dx**1.5
        density = np.abs(Zphi / norm_factor)**2

        # 3种平面投影
        INTZ = np.sum(density, axis=2) * dx
        INTX = np.sum(density, axis=0) * dx
        INTY = np.sum(density, axis=1) * dx

        # ---------- Plot XY plane (Z integrated) ----------
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(INTZ, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='viridis')
        ax.set_title(f'{state_labels[idx]} XY Density (E = {energy:.8f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"{state_labels[idx]}_XY_plane_N{N}.pdf")
        plt.close()

        # ---------- Plot XZ plane (Y integrated) ----------
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(INTY, extent=(X.min(), X.max(), Z.min(), Z.max()), origin='lower', cmap='plasma')
        ax.set_title(f'{state_labels[idx]} XZ Density (E = {energy:.8f})')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"{state_labels[idx]}_XZ_plane_N{N}.pdf")
        plt.close()

        # ---------- Plot YZ plane (X integrated) ----------
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(INTX, extent=(Y.min(), Y.max(), Z.min(), Z.max()), origin='lower', cmap='inferno')
        ax.set_title(f'{state_labels[idx]} YZ Density (E = {energy:.8f})')
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"{state_labels[idx]}_YZ_plane_N{N}.pdf")
        plt.close()


