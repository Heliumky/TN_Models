import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CytnxTools')))
import matplotlib.pyplot as plt
import pickle
import plot_utility_jit as ptut
import npmps
import numpy as np


input_mps_path = "2d_dmrg_N=5.pkl"

with open(input_mps_path, 'rb') as file:
    loaded_data = pickle.load(file)

N = loaded_data['N']
shift = loaded_data['shift']
rescale = loaded_data['rescale']
xmax = loaded_data['xmax']
cutoff = loaded_data['cutoff']
dx = loaded_data['dx']
Energy = loaded_data['Energy']
phi1 = loaded_data['phi1']
phi2 = loaded_data['phi2']
psi = loaded_data['psi']

print(psi[0].shape)

#phi0 = npmps.kill_site(phi0, sysdim=2, dtype=phi0[0].dtype)
#phi1 = npmps.kill_site(phi1, sysdim=2, dtype=phi1[0].dtype)
#print(npmps.inner_MPS(phi0,phi0))



#phi0 = npmps.kill_site(phi0, sysdim=2, dtype=phi0[0].dtype)
#phi1 = npmps.kill_site(phi1, sysdim=2, dtype=phi1[0].dtype)

'''

# ---------- Grid and Mapping ----------
bxs = list(ptut.BinaryNumbers(N+1))
bys = list(ptut.BinaryNumbers(N+1))
xs = ptut.bin_to_dec_list(bxs, rescale=rescale, shift=shift)
ys = ptut.bin_to_dec_list(bys, rescale=rescale, shift=shift)
X, Y = np.meshgrid(xs, ys)
phis_grow = [npmps.grow_site_0th(phi, sysdim=2, dtype=float) for phi in phis]

wfs = [ptut.get_2D_mesh_eles_mps(phi, bxs, bys) for phi in phis_grow]

# ---------- Convert MPS to Grid ----------
wfs = [ptut.get_2D_mesh_eles_mps(phi, bxs, bys) for phi in phis_grow]
#ZV = ptut.get_2D_mesh_eles_mps(V_MPS, bxs, bys)
titles = ['1s', '2p', '2p', '2s']
cmaps = ['plasma'] * 4



plt.figure(figsize=(16, 8))
for i, (wf, title, cmap) in enumerate(zip(wfs, titles, cmaps)):
    # Real part
    plt.subplot(2, 4, i + 1)
    plt.imshow(wf.real / dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$Re(\psi)$')
    plt.title(f'{title} Real Part\nEnergy = {energy_[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')

    # psi2
    plt.subplot(2, 4, i + 5)
    plt.imshow(np.abs(wf / dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$|\psi|^2$')
    plt.title(f'{title} Density\nEnergy = {energy_[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')
plt.tight_layout()
plt.savefig(f"2d_extrp_{N}.pdf", format='pdf')
plt.show()
'''
