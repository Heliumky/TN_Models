import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
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

prob_density = np.abs(Z1 / dx**1.5)**2

from qmsolve import init_visualization

# 计算波函数而非概率密度
psi = Z1 / dx**1.5 

fs_density = (Z1 / dx**1.5)**2 

class FakeEigenstates:
    def __init__(self, psi, energies, extent, N, number, dtype):
        self.array = np.array([psi])  # 波函数数组必须是3D复数
        self.energies = energies         # 必须存在能量列表
        self.extent = extent          # 必须为6元素列表 [xmin, xmax, ...]
        self.N = N         # 建议保留网格维度
        self.number = number
        self.type = str(dtype)


eigenstates = FakeEigenstates(fs_density, energy_2s, max(xs)*2, psi.shape[0], 1, 'SingleParticle3D')
print(eigenstates.energies)



visualization = init_visualization(eigenstates)

#visualization.plot_eigenstate(1, contrast_vals = [0.01, 0.2])
#visualization.animate(contrast_vals = [0.01, 0.2])

