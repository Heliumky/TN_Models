import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../CytnxTools')))
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



input_mps_path = "2d_dmrg_N=9_par=0.pkl"

with open(input_mps_path, 'rb') as file:
    loaded_data = pickle.load(file)

Rad = loaded_data['Rad']
parity = loaded_data['parity']
N = loaded_data['N']
shift = loaded_data['shift']
rescale = loaded_data['rescale']
xmax = loaded_data['xmax']
cutoff = loaded_data['cutoff']
dx = loaded_data['dx']
Energy = loaded_data['Energy'] + 1/Rad
phi1 = loaded_data['phi1']
phi2 = loaded_data['phi2']
psi = loaded_data['psi']
occ1 = loaded_data['occ1']
occ2 = loaded_data['occ2']

# --------------- Plot ---------------------
bxs = list(ptut.BinaryNumbers(N))
bys = list(ptut.BinaryNumbers(N))

xs = ptut.bin_to_dec_list (bxs,rescale=rescale, shift=shift)
ys = ptut.bin_to_dec_list (bys,rescale=rescale, shift=shift)
X, Y = np.meshgrid (xs, ys)


# ---------- Convert MPS to Grid ----------
phis = [phi1, phi2]
wavefunctions  = [ptut.get_2D_mesh_eles_mps(phi, bxs, bys) for phi in phis]

plt.figure(figsize=(8, 8))

titles = [f'phi1_par={parity}', f'phi2_par={parity}']
cmaps = ['viridis', 'plasma']
occ_arr = [occ1, occ2]
# ---------- Plot in Matplotlib ----------
for i, (Z, title, cmap) in enumerate(zip(wavefunctions, titles, cmaps)):
    # 实部
    plt.subplot(2, 2, i + 1)
    plt.imshow(Z.real/dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$Re(\psi)$')
    plt.title(f'{title} Real Part\nEnergy = {Energy:.8f} \nOBOC={occ_arr[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2, 2, i + 3)
    plt.imshow(np.abs(Z/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$|\psi|^2$')
    plt.title(f'{title} Density\nEnergy = {Energy:.8f} \nOBOC={occ_arr[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')
plt.tight_layout()
plt.savefig(f"2d_dmrg_{N}_par={parity}.pdf", format='pdf')
plt.show()
