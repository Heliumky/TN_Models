import plot_utility as ptut
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import plotsetting as ps
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, MaxNLocator

input_mps_path = "2d_dmrg_results_N=9.pkl"
with open(input_mps_path, 'rb') as file:
    loaded_data = pickle.load(file)
#print(loaded_data)
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



# --------------- Plot ---------------------
bxs = list(ptut.BinaryNumbers(N))
bys = list(ptut.BinaryNumbers(N))

xs = ptut.bin_to_dec_list (bxs,rescale=rescale, shift=shift)
ys = ptut.bin_to_dec_list (bys,rescale=rescale, shift=shift)
X, Y = np.meshgrid (xs, ys)



#ZV = ptut.get_3D_mesh_eles_mps (V_MPS, bxs, bys, bzs)
Z0 = ptut.get_2D_mesh_eles_mps (phi0, bxs, bys)
Z1 = ptut.get_2D_mesh_eles_mps (phi1, bxs, bys)


#1s plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
surfxy0 = ax[0].imshow(np.abs(Z0/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), 
                       origin='lower', cmap='viridis')

ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_xlabel(r'$x$', rotation=0, fontsize=25)
ax[0].set_ylabel(r'$y$', rotation=0, fontsize=25)
ax[0].set_aspect('equal', adjustable='box')
ax[0].set_title(f'1s Density (Energy = {energy_1s:.8f})')
ax[0].set_xlim(shift, -shift)
ax[0].set_ylim(shift, -shift)
ax[0].text(0.1, 0.9, "(a)", fontsize=25, transform=ax[0].transAxes)
ps.set_tick_inteval(ax[0].yaxis, major_itv=4, minor_itv=2)
ps.set_tick_inteval(ax[0].xaxis, major_itv=4, minor_itv=2)
ps.set(ax[0])
cbar0 = fig.colorbar(surfxy0, ax=ax[0])
cbar0.ax.tick_params(labelsize=25)


#2s plotting
surfxy1 = ax[1].imshow(np.abs(Z1/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), 
                       origin='lower', cmap='plasma')
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set_xlabel(r'$x$', rotation=0, fontsize=25)
ax[1].set_ylabel(r'$y$', rotation=0, fontsize=25)
ax[1].set_aspect('equal', adjustable='box')
ax[1].set_title(f'2s Density (Energy = {energy_2s:.8f})')
ax[1].set_xlim(shift, -shift)
ax[1].set_ylim(shift, -shift)
ax[1].text(0.1, 0.9, "(b)", fontsize=25, transform=ax[1].transAxes)
ps.set_tick_inteval(ax[1].yaxis, major_itv=4, minor_itv=2)
ps.set_tick_inteval(ax[1].xaxis, major_itv=4, minor_itv=2)
ps.set(ax[1])
cbar1 = fig.colorbar(surfxy1, ax=ax[1])
cbar1.ax.tick_params(labelsize=25)
plt.tight_layout()

# save fig
plt.savefig(f"2d_dmrg_density_site_{N}_deg.pdf", format='pdf')

plt.show()

