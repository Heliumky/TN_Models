import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
import tci
import plot_utility_jit as ptut
import matplotlib.pyplot as plt
import pickle
import numpy as np
from mayavi import mlab

import matplotlib.pyplot as plt
from mayavi import mlab

# This function is from the "https://github.com/quantum-visualizations/qmsolve/blob/main/qmsolve/visualization/single_particle_3D.py" 

def plot_eigenstate(psi, X, Y, Z, plot_type, contrast_vals=[0.1, 0.25], filename="mayavi_plot.png"):
    """
    Plot 3D eigenstate using Mayavi and save it as an image via Matplotlib screenshot.
    """
    mlab.options.offscreen = True  # if false, it make the render problems
    psi = psi / np.abs(np.max(psi))

    if plot_type == 'density':
        data = np.abs(psi) ** 2
    elif plot_type == 'real_part':
        data = np.real(psi)
    elif plot_type == 'imag_part':
        data = np.imag(psi)
    else:
        raise ValueError("plot_type must be one of: 'density', 'real_part', 'imag_part'")

    # Create Mayavi figure
    fig = mlab.figure(size=(2400, 2400), bgcolor=(1, 1, 1)) 
    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(data)) 
    mlab.outline(color=(0, 0, 0)) # black color for bolder 
    axes = mlab.axes(xlabel='x', ylabel='y', zlabel='z', nb_labels=6, ranges=(X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max()), color=(0, 0, 0)) 
    mlab.view(azimuth=30, distance=len(X) * 3.5)
    axes.label_text_property.color = (0, 0, 0) # Black color for axes
    axes.title_text_property.color = (0, 0, 0) # label black color
    
    # Force rendering
    mlab.draw()
    fig.scene.render()

    # **Matplotlib captures screenshot**
    plt.figure(figsize=(12, 12))
    plt.imshow(mlab.screenshot(antialiased=True))  # High-quality screenshot
    plt.axis("off")  # Hide Matplotlib axes
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save high-resolution PNG
    plt.close()
    mlab.close()


# ---------- Load Data ----------
input_mps_path = "3d_dmrg_results_N=6.pkl"
with open(input_mps_path, 'rb') as file:
    loaded_data = pickle.load(file)

N = loaded_data['N']
shift = loaded_data['shift']
rescale = loaded_data['rescale']
dx = loaded_data['dx']
energy_ = loaded_data['energy_arr']
phis = loaded_data['phi_arr']

# ---------- Grid and Mapping ----------
bxs = list(ptut.BinaryNumbers(N))
bys = list(ptut.BinaryNumbers(N))
bzs = list(ptut.BinaryNumbers(N))
xs = ptut.bin_to_dec_list(bxs, rescale=rescale, shift=shift)
ys = ptut.bin_to_dec_list(bys, rescale=rescale, shift=shift)
zs = ptut.bin_to_dec_list(bzs, rescale=rescale, shift=shift)
X, Y, Z = np.meshgrid(xs, ys, zs)

# ---------- Convert MPS to Grid ----------
wfs = [ptut.get_3D_mesh_eles_mps(phi, bxs, bys, bzs) for phi in phis]
titles = ['1s', '2px', '2py', '2pz', '2s']

# ---------- Plot in Matplotlib ----------
fig, axes = plt.subplots(2, 5, figsize=(16, 8))

for i, (wf, title) in enumerate(zip(wfs, titles)):
    # Mayavi 
    filename = f"mayavi_plot_density{i}.png"
    plot_eigenstate(np.abs(wf / (dx)**1.5), X, Y, Z, 'density', contrast_vals=[0.01, 0.5], filename=filename)
    #  Matplotlib subplot 
    mayavi_img = plt.imread(filename)
    axes[0, i].imshow(mayavi_img)
    axes[0, i].axis("off")
    axes[0, i].set_title(f"{title} Density\nEnergy = {energy_[i]:.8f}")
    
    filename = f"mayavi_plot_wf{i}.png"
    plot_eigenstate(wf / (dx)**1.5, X, Y, Z, 'real_part', contrast_vals=[0.01, 0.5], filename=filename)
    mayavi_img = plt.imread(filename)
    axes[1, i].imshow(mayavi_img)
    axes[1, i].axis("off")
    axes[1, i].set_title(f"{title} Real part\nEnergy = {energy_[i]:.8f}")
plt.tight_layout()
plt.savefig(f"3d_dmrg_{N}.pdf", format='pdf', bbox_inches='tight')
plt.show()

