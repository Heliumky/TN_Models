import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
import tci
import plot_utility_jit as ptut
import matplotlib.pyplot as plt
import pickle
import os
#import plotsetting as ps
import matplotlib.pyplot as plt
#from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, MaxNLocator


input_mps_path = "2d_dmrg_results_N=10.pkl"

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
V_MPS = tci.load_mps(f'fit{2*N}.mps.npy')
Z2 = ptut.get_2D_mesh_eles_mps (V_MPS, bxs, bys)


plt.figure(figsize=(8, 12))
plt.rcParams.update({
    'font.size': 30,             # 全局字體大小
    'axes.titlesize': 26,        # 子圖標題大小
    'axes.labelsize': 26,        # x, y 標籤字體
    'xtick.labelsize': 24,       # x 軸刻度字體
    'ytick.labelsize': 24,       # y 軸刻度字體
    'legend.fontsize': 24,       # 圖例字體
    'colorbar.labelsize': 24     # colorbar 字體
})
wavefunctions = [Z0, Z1]
eigenvalues = [energy_1s, energy_2s]
titles = ['1s', '2?']
cmaps = ['viridis', 'plasma']

for i, (Z, title, cmap) in enumerate(zip(wavefunctions, titles, cmaps)):
    # 实部
    plt.subplot(3, 2, i + 1)
    plt.imshow(Z.real/dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$Re(\psi)$')
    plt.title(f'{title} Real Part\nEnergy = {eigenvalues[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')

    # 虚部
    plt.subplot(3, 2, i + 3)
    plt.imshow(Z.imag/dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$Im(\psi)$')
    plt.title(f'{title} Imag Part\nEnergy = {eigenvalues[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(3, 2, i + 5)
    plt.imshow(np.abs(Z/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$|\psi|^2$')
    plt.title(f'{title} Density\nEnergy = {eigenvalues[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')
plt.tight_layout()
plt.savefig(f"2d_dmrg_{N}.pdf", format='pdf')
plt.show()



