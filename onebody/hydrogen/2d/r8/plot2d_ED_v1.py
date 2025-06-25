import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CytnxTools')))
import tci
import plot_utility_jit as ptut
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------- Load Data ----------
input_mps_path = "2d_ED_results_N=8.pkl"

with open(input_mps_path, 'rb') as file:
    loaded_data = pickle.load(file)

N        = loaded_data['N']
shift    = loaded_data['shift']
rescale  = loaded_data['rescale']
xmax     = loaded_data['xmax']
cutoff   = loaded_data['cutoff']
dx       = loaded_data['dx']
eigvals  = loaded_data['eigvals']
mps_arr  = loaded_data['mps_arr']
V_MPS    = loaded_data['V_MPS']

# ---------- Prepare Grid ----------
bxs = list(ptut.BinaryNumbers(N))
bys = list(ptut.BinaryNumbers(N))
xs = ptut.bin_to_dec_list(bxs, rescale=rescale, shift=shift)
ys = ptut.bin_to_dec_list(bys, rescale=rescale, shift=shift)
X, Y = np.meshgrid(xs, ys)

Z0 = ptut.get_2D_mesh_eles_mps(mps_arr[0], bxs, bys)
Z1 = ptut.get_2D_mesh_eles_mps(mps_arr[1], bxs, bys)
Z2 = ptut.get_2D_mesh_eles_mps(mps_arr[2], bxs, bys)
Z3 = ptut.get_2D_mesh_eles_mps(mps_arr[3], bxs, bys)

wavefunctions = [Z0, Z1, Z2, Z3]
titles = ['1s', '2p', '2p', '2s']

# ---------- Plot ----------
fig = plt.figure(figsize=(18, 9))
gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], hspace=0.05, wspace=0.05)

# 自动计算 vmin/vmax
re_min = min((Z/dx).real.min() for Z in wavefunctions)
re_max = max((Z/dx).real.max() for Z in wavefunctions)
dens_min = 0
dens_max = max((np.abs(Z / dx)**2).max() for Z in wavefunctions)

# --- First Row: Real parts ---
for i, (Z, title) in enumerate(zip(wavefunctions, titles)):
    ax = plt.subplot(gs[0, i])
    im = ax.imshow(Z.real / dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower',
                   cmap='turbo', vmin=re_min, vmax=re_max)
    if i == 0:
        ax.set_ylabel('y', fontsize=24)
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(labelsize=22)
    ax.set_title(f'{title} $Re(\\psi)$\nE = {eigvals[i]:.8f}', fontsize=24)

# --- Colorbar for Re(ψ) ---
cax1 = plt.subplot(gs[0, -1])
cbar1 = fig.colorbar(im, cax=cax1)
cbar1.set_label(r'$Re(\psi)$', fontsize=22)
cbar1.ax.tick_params(labelsize=18)

# --- Second Row: Densities ---
for i, (Z, title) in enumerate(zip(wavefunctions, titles)):
    ax = plt.subplot(gs[1, i])
    im2 = ax.imshow(np.abs(Z / dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower',
                    cmap='turbo', vmin=dens_min, vmax=dens_max)
    if i == 0:
        ax.set_ylabel('y', fontsize=24)
    else:
        ax.set_yticks([])
    ax.set_xlabel('x', fontsize=24)
    ax.tick_params(labelsize=22)

# --- Colorbar for |ψ|² ---
cax2 = plt.subplot(gs[1, -1])
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.set_label(r'$|\psi|^2$', fontsize=22)
cbar2.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig(f"2d_extdiag_compact_{N}_sharedcolorbar.pdf", format='pdf')
plt.show()

