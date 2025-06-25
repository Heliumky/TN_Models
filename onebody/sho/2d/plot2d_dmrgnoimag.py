import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../CytnxTools')))
import tci
import plot_utility_jit as ptut
import matplotlib.pyplot as plt
import pickle
import numpy as np

# ---------- Load Data ----------
input_mps_path = "2d_dmrg_results_N=8.pkl"
with open(input_mps_path, 'rb') as file:
    loaded_data = pickle.load(file)

N = loaded_data['N']
shift = loaded_data['shift']
rescale = loaded_data['rescale']
xmax = loaded_data['xmax']
cutoff = loaded_data['cutoff']
dx = loaded_data['dx']
energy_ = loaded_data['energy_arr']
phis = loaded_data['phi_arr']
#V_MPS = loaded_data['V_MPS']

# ---------- Grid and Mapping ----------
bxs = list(ptut.BinaryNumbers(N))
bys = list(ptut.BinaryNumbers(N))
xs = ptut.bin_to_dec_list(bxs, rescale=rescale, shift=shift)
ys = ptut.bin_to_dec_list(bys, rescale=rescale, shift=shift)
X, Y = np.meshgrid(xs, ys)

# 找最接近 x=0 和 y=0 的索引（給切面用）
ix0 = np.argmin(np.abs(xs))
iy0 = np.argmin(np.abs(ys))

# ---------- Convert MPS to Grid ----------
wfs = [ptut.get_2D_mesh_eles_mps(phi, bxs, bys) for phi in phis]
#ZV = ptut.get_2D_mesh_eles_mps(V_MPS, bxs, bys)
titles = ['1s', '2p', '2p', '2s']
cmaps = ['plasma'] * 4

# ---------- Plot 2D Maps ----------
plt.figure(figsize=(16, 8))
for i, (wf, title, cmap) in enumerate(zip(wfs, titles, cmaps)):
    # Real part
    plt.subplot(2, 4, i + 1)
    plt.imshow(wf.real / dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$Re(\psi)$')
    plt.title(f'{title} Real Part\nEnergy = {energy_[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')

    # |ψ|^2
    plt.subplot(2, 4, i + 5)
    plt.imshow(np.abs(wf / dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$|\psi|^2$')
    plt.title(f'{title} Density\nEnergy = {energy_[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')
plt.tight_layout()
plt.savefig(f"2d_dmrg_{N}.pdf", format='pdf')
plt.show()

# ---------- Plot Cross Sections ----------
for i, (wf, title) in enumerate(zip(wfs, titles)):
    plt.figure(figsize=(10, 4))

    # y=0 cut (horizontal)
    plt.subplot(1, 2, 1)
    plt.plot(xs, wf.real[iy0, :] / dx, label=f'$Re(\psi)$')
    plt.plot(xs, (np.abs(wf[iy0, :] / dx))**2, label=f'$|\psi|^2$', linestyle='--')
    plt.title(f'{title} @ y=0')
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.legend()

    # x=0 cut (vertical)
    plt.subplot(1, 2, 2)
    plt.plot(ys, wf.real[:, ix0] / dx, label=f'$Re(\psi)$')
    plt.plot(ys, (np.abs(wf[:, ix0] / dx))**2, label=f'$|\psi|^2$', linestyle='--')
    plt.title(f'{title} @ x=0')
    plt.xlabel('y')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"2d_dmrg_cut_{title}_N{N}.pdf", format='pdf')
    plt.close()

