import os, sys
sys.path.append(os.path.abspath('../CytnxTools'))
import matplotlib.pyplot as plt
import pickle
import plot_utility_jit as ptut
import npmps
import numpy as np

input_mps_path = "3d_dmrg_results_N=5.pkl"
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
bzs = list(ptut.BinaryNumbers(N))


xs = ptut.bin_to_dec_list(bxs, rescale=dx, shift=shift)
ys = ptut.bin_to_dec_list(bys, rescale=dx, shift=shift)
zs = ptut.bin_to_dec_list(bzs, rescale=dx, shift=shift)
print(xs[0])
X, Y, Z = np.meshgrid(xs, ys, zs)


Z0 = ptut.get_3D_mesh_eles_mps(phi0, bxs, bys, bzs)
Z1 = ptut.get_3D_mesh_eles_mps(phi1, bxs, bys, bzs)

from qmsolve import Eigenstates, init_visualization,Å, nm

xs = ptut.bin_to_dec_list(bxs, rescale=dx, shift=shift)
ys = ptut.bin_to_dec_list(bys, rescale=dx, shift=shift)
zs = ptut.bin_to_dec_list(bzs, rescale=dx, shift=shift)



# density
fs_density = np.abs(Z1 / (dx/Å)**1.5)**2
gs_density = np.abs(Z0 / (dx/Å)**1.5)**2

# construct Eigenstates
#eigenstates = Eigenstates(np.array([energy_1s,energy_2s]), np.array([gs_density ,fs_density]), 2*max(xs)*Å, fs_density.shape[0], "SingleParticle3D")
eigenstates = Eigenstates(energy_2s, np.array([fs_density]), 2*max(xs)*Å, fs_density.shape[0], "SingleParticle3D")
visualization = init_visualization(eigenstates)
visualization.plot_eigenstate(0, contrast_vals = [0.01, 0.5])
visualization.plot_eigenstate(1, contrast_vals = [0.01, 0.5])
import inspect
print(inspect.signature(init_visualization(eigenstates)))
print(inspect.signature(visualization.plot_eigenstate))
#visualization.animate(contrast_vals =  [0.01, 0.5])
