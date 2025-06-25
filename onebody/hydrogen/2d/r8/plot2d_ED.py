import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CytnxTools')))
import tci
import plot_utility_jit as ptut
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, MaxNLocator


input_mps_path = "2d_ED_results_N=5.pkl"

with open(input_mps_path, 'rb') as file:
    loaded_data = pickle.load(file)
#print(loaded_data)
N = loaded_data['N']
shift = loaded_data['shift']
rescale = loaded_data['rescale']
xmax = loaded_data['xmax']
cutoff = loaded_data['cutoff']
dx = loaded_data['dx']
eigvals = loaded_data['eigvals']
mps_arr = loaded_data['mps_arr']
V_MPS = loaded_data['V_MPS']

# --------------- Plot ---------------------

bxs = list(ptut.BinaryNumbers(N))
bys = list(ptut.BinaryNumbers(N))

xs = ptut.bin_to_dec_list (bxs,rescale=rescale, shift=shift)
ys = ptut.bin_to_dec_list (bys,rescale=rescale, shift=shift)
X, Y = np.meshgrid (xs, ys)

ZV = ptut.get_2D_mesh_eles_mps (V_MPS, bxs, bys)
Z0 = ptut.get_2D_mesh_eles_mps (mps_arr[0], bxs, bys)
Z1 = ptut.get_2D_mesh_eles_mps (mps_arr[1], bxs, bys)
Z2 = ptut.get_2D_mesh_eles_mps (mps_arr[2], bxs, bys)
Z3 = ptut.get_2D_mesh_eles_mps (mps_arr[3], bxs, bys)

plt.figure(figsize=(16, 8))

wavefunctions = [Z0, Z1, Z2, Z3]
titles = ['1s', '2p', '2p', '2s']
cmaps = ['plasma', 'plasma', 'plasma', 'plasma']

for i, (Z, title, cmap) in enumerate(zip(wavefunctions, titles, cmaps)):
    # 实部
    plt.subplot(2, 4, i + 1)
    plt.imshow(Z.real/dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$Re(\psi)$')
    plt.title(f'{title} Real Part\nEnergy = {eigvals[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')

    # 虚部
    #plt.subplot(3, 4, i + 5)
    #plt.imshow(Z.imag/dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    #plt.colorbar(label=f'$Im(\psi)$')
    #plt.title(f'{title} Imag Part\nEnergy = {eigvals[i]:.8f}')
    #plt.xlabel('x')
    #plt.ylabel('y')

    #plt.subplot(3, 4, i + 9)
    plt.subplot(2, 4, i + 5)
    plt.imshow(np.abs(Z/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$|\psi|^2$')
    plt.title(f'{title} Density\nEnergy = {eigvals[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')
plt.tight_layout()
plt.savefig(f"2d_extdiag_{N}.pdf", format='pdf')
plt.show()

