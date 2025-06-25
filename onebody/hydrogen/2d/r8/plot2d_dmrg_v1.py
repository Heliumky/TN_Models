import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CytnxTools')))
import tci
import plot_utility_jit as ptut
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
energy_1s = loaded_data['energy_1s']
energy_2s = loaded_data['energy_2s']
phi0 = loaded_data['phi0']
phi1 = loaded_data['phi1']

bxs = list(ptut.BinaryNumbers(N))
bys = list(ptut.BinaryNumbers(N))

xs = ptut.bin_to_dec_list(bxs, rescale=rescale, shift=shift)
ys = ptut.bin_to_dec_list(bys, rescale=rescale, shift=shift)
X, Y = np.meshgrid(xs, ys)

Z0 = ptut.get_2D_mesh_eles_mps(phi0, bxs, bys)
Z1 = ptut.get_2D_mesh_eles_mps(phi1, bxs, bys)

wavefunctions = [Z0, Z1]
titles = ['1s', '2s']
energies = [energy_1s, energy_2s]

# ---------- Plot ----------
fig = plt.figure(figsize=(14, 18))
gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 0.05], hspace=0.00, wspace=0.01)
#fig = plt.figure(figsize=(6, 24), constrained_layout=True)
#gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 0.05], wspace=0.01, hspace=0.03)
# Compute vmin/vmax for colorbars
re_min = min((Z/dx).real.min() for Z in wavefunctions)
re_max = max((Z/dx).real.max() for Z in wavefunctions)
im_min = min((Z/dx).imag.min() for Z in wavefunctions)
im_max = max((Z/dx).imag.max() for Z in wavefunctions)
dens_min = 0
dens_max = max((np.abs(Z / dx)**2).max() for Z in wavefunctions)

# Row 0: Real part
for i, (Z, title, energy) in enumerate(zip(wavefunctions, titles, energies)):
    ax = plt.subplot(gs[0, i])
    im_re = ax.imshow(Z.real / dx, extent=(X.min(), X.max(), Y.min(), Y.max()),
                      origin='lower', cmap='turbo', vmin=re_min, vmax=re_max)
    if i == 0:
        ax.set_ylabel('y', fontsize=24)
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(labelsize=22)
    ax.set_title(f'{title} \nE = {energy:.8f}', fontsize=24)

cax_re = plt.subplot(gs[0, -1])
cbar_re = fig.colorbar(im_re, cax=cax_re)
cbar_re.set_label(r'$Re(\psi)$', fontsize=22)
cbar_re.ax.tick_params(labelsize=22)

# Row 1: Imag part
for i, (Z, title, energy) in enumerate(zip(wavefunctions, titles, energies)):
    ax = plt.subplot(gs[1, i])
    im_im = ax.imshow(Z.imag / dx, extent=(X.min(), X.max(), Y.min(), Y.max()),
                      origin='lower', cmap='turbo', vmin=im_min, vmax=im_max)
    if i == 0:
        ax.set_ylabel('y', fontsize=24)
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(labelsize=22)
    #ax.set_title(f'{title} $Im(\\psi)$', fontsize=24)

cax_im = plt.subplot(gs[1, -1])
cbar_im = fig.colorbar(im_im, cax=cax_im)
cbar_im.set_label(r'$Im(\psi)$', fontsize=22)
cbar_im.ax.tick_params(labelsize=22)

# Row 2: Density
for i, (Z, title, energy) in enumerate(zip(wavefunctions, titles, energies)):
    ax = plt.subplot(gs[2, i])
    im_dens = ax.imshow(np.abs(Z / dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()),
                        origin='lower', cmap='turbo', vmin=dens_min, vmax=dens_max)
    if i == 0:
        ax.set_ylabel('y', fontsize=24)
    else:
        ax.set_yticks([])
    ax.set_xlabel('x', fontsize=24)
    ax.tick_params(labelsize=22)

cax_dens = plt.subplot(gs[2, -1])
cbar_dens = fig.colorbar(im_dens, cax=cax_dens)
cbar_dens.set_label(r'$|\psi|^2$', fontsize=22)
cbar_dens.ax.tick_params(labelsize=22)

plt.tight_layout()
plt.savefig(f"2d_dmrg_full_{N}_with_imag.pdf", format='pdf')
plt.show()

