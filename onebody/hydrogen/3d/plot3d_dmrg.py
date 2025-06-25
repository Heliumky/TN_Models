import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
import plot_utility_jit as ptut
import matplotlib.pyplot as plt
import pickle
import numpy as np
from mayavi import mlab
import numpy as np
from tvtk.util import ctf

def plot_eigenstate(psi, X, Y, Z, plot_type, contrast_vals=[0.1, 0.25]):
    """
    Plot 3D eigenstate using Mayavi.
    
    Parameters:
        psi           : 3D complex array, wavefunction
        x, y, z       : 1D arrays representing coordinates
        plot_type     : 'density', 'real_part', or 'imag_part'
        contrast_vals : [vmin, vmax] for color scaling
    """
    # Generate grid

    psi = psi/np.abs(np.max(psi))
    if plot_type == 'density':
        data = np.abs(psi) ** 2
    elif plot_type == 'real_part':
        data = np.real(psi)
    elif plot_type == 'imag_part':
        data = np.imag(psi)
    else:
        raise ValueError("plot_type must be one of: 'density', 'real_part', 'imag_part'")

    # Plot setup
    N = psi.shape[0]
    #mlab.figure(bgcolor=(0, 0, 0), size=(700, 700))
    
    vol = mlab.pipeline.volume(
        mlab.pipeline.scalar_field(data)
    )
    #if plot_type == 'density':
    c = ctf.save_ctfs(vol._volume_property)
    c['rgb'] = [[-0.45, 0.3, 0.3, 1.0],
                [-0.4, 0.1, 0.1, 1.0],
                [-0.3, 0.0, 0.0, 1.0],
                [-0.2, 0.0, 0.0, 1.0],
                [-0.001, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.001, 1.0, 0.0, 0.0],
                [0.2, 1.0, 0.0, 0.0],
                [0.3, 1.0, 0.0, 0.0],
                [0.4, 1.0, 0.1, 0.1],
                [0.45, 1.0, 0.3, 0.3]]
    c['alpha'] = [[-0.5, 1.0],
                  [-contrast_vals[1], 1.0],
                  [-contrast_vals[0], 0.0],
                  [0, 0.0],
                  [contrast_vals[0], 0.0],
                  [contrast_vals[1], 1.0],
                  [0.5, 1.0]]
    ctf.load_ctfs(c, vol._volume_property)
    vol.update_ctf = True
    # If density, add color transfer function
    # Axes and view
    mlab.outline()
    mlab.axes(
        xlabel='x', ylabel='y', zlabel='z', nb_labels=6,
        ranges=(X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max())
    )
    mlab.view(azimuth=30, distance=N * 3.5)
    mlab.show()


input_mps_path = "3d_dmrg_results_N=8.pkl"
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

xs = ptut.bin_to_dec_list(bxs, rescale=rescale, shift=shift)
ys = ptut.bin_to_dec_list(bys, rescale=rescale, shift=shift)
zs = ptut.bin_to_dec_list(bzs, rescale=rescale, shift=shift)
X, Y, Z = np.meshgrid(xs, ys, zs)

Z0 = ptut.get_3D_mesh_eles_mps(phi0, bxs, bys, bzs)
Z1 = ptut.get_3D_mesh_eles_mps(phi1, bxs, bys, bzs)

plot_eigenstate(np.abs(Z0 / (dx)**1.5), X, Y, Z, 'density', contrast_vals = [0.01, 0.5])
plot_eigenstate(np.abs(Z1 / (dx)**1.5), X, Y, Z, 'density', contrast_vals = [0.01, 0.5])


INTZ_Z0 = np.sum(np.abs(Z0 / dx**1.5)**2, axis=2) * dx
INTZ_Z1 = np.sum(np.abs(Z1 / dx**1.5)**2, axis=2) * dx
INTX_Z0 = np.sum(np.abs(Z0 / dx**1.5)**2, axis=0) * dx
INTX_Z1 = np.sum(np.abs(Z1 / dx**1.5)**2, axis=0) * dx
INTY_Z0 = np.sum(np.abs(Z0 / dx**1.5)**2, axis=1) * dx
INTY_Z1 = np.sum(np.abs(Z1 / dx**1.5)**2, axis=1) * dx

# XY 
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(INTZ_Z0, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='viridis')
axs[0].set_title(f'1s XY Density (Energy = {energy_1s:.8f})')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].text(0.1, 0.9, "(a)", fontsize=25, transform=axs[0].transAxes)
plt.colorbar(axs[0].imshow(INTZ_Z0, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='viridis'), ax=axs[0])

axs[1].imshow(INTZ_Z1, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='plasma')
axs[1].set_title(f'2s XY Density (Energy = {energy_2s:.8f})')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].text(0.1, 0.9, "(b)", fontsize=25, transform=axs[1].transAxes)
plt.colorbar(axs[1].imshow(INTZ_Z1, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='plasma'), ax=axs[1])

plt.tight_layout()
plt.savefig("XY_plane.pdf", format='pdf')

# XZ 
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(INTY_Z0, extent=(X.min(), X.max(), Z.min(), Z.max()), origin='lower', cmap='viridis')
axs[0].set_title(f'1s XZ Density (Energy = {energy_1s:.8f})')
axs[0].set_xlabel('x')
axs[0].set_ylabel('z')
axs[0].text(0.1, 0.9, "(c)", fontsize=25, transform=axs[0].transAxes)
plt.colorbar(axs[0].imshow(INTY_Z0, extent=(X.min(), X.max(), Z.min(), Z.max()), origin='lower', cmap='viridis'), ax=axs[0])

axs[1].imshow(INTY_Z1, extent=(X.min(), X.max(), Z.min(), Z.max()), origin='lower', cmap='plasma')
axs[1].set_title(f'2s XZ Density (Energy = {energy_2s:.8f})')
axs[1].set_xlabel('x')
axs[1].set_ylabel('z')
axs[1].text(0.1, 0.9, "(d)", fontsize=25, transform=axs[1].transAxes)
plt.colorbar(axs[1].imshow(INTY_Z1, extent=(X.min(), X.max(), Z.min(), Z.max()), origin='lower', cmap='plasma'), ax=axs[1])

plt.tight_layout()
plt.savefig("XZ_plane.pdf", format='pdf')

# YZ 
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(INTX_Z0, extent=(Y.min(), Y.max(), Z.min(), Z.max()), origin='lower', cmap='viridis')
axs[0].set_title(f'1s YZ Density (Energy = {energy_1s:.8f})')
axs[0].set_xlabel('y')
axs[0].set_ylabel('z')
axs[0].text(0.1, 0.9, "(e)", fontsize=25, transform=axs[0].transAxes)
plt.colorbar(axs[0].imshow(INTX_Z0, extent=(Y.min(), Y.max(), Z.min(), Z.max()), origin='lower', cmap='viridis'), ax=axs[0])

axs[1].imshow(INTX_Z1, extent=(Y.min(), Y.max(), Z.min(), Z.max()), origin='lower', cmap='plasma')
axs[1].set_title(f'2s YZ Density (Energy = {energy_2s:.8f})')
axs[1].set_xlabel('y')
axs[1].set_ylabel('z')
axs[1].text(0.1, 0.9, "(f)", fontsize=25, transform=axs[1].transAxes)
plt.colorbar(axs[1].imshow(INTX_Z1, extent=(Y.min(), Y.max(), Z.min(), Z.max()), origin='lower', cmap='plasma'), ax=axs[1])

plt.tight_layout()
plt.savefig("YZ_plane.pdf", format='pdf')


