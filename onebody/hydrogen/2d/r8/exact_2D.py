import numpy as np
import matplotlib.pyplot as plt
import plotsetting as ps
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, MaxNLocator



# meshgrid
N = 9
shift = -8

x = np.linspace(shift, -shift, 2**N)
y = np.linspace(shift, -shift, 2**N)
X, Y = np.meshgrid(x, y)  

# potential
V = -1 / np.sqrt(X**2 + Y**2) 
#V = 0.5*(X**2 + Y**2)
# polar coordinate
r = np.sqrt(X**2 + Y**2)

plt.figure(figsize=(12, 6))
plt.plot(x, -1 / np.sqrt(x**2))

# ground state 2D hydrogen：psi = exp(-2r)
psi_ground = np.exp(-2 * r)
dx = x[1] - x[0]
dy = y[1] - y[0]
print(dx)

def second_derivative_nd(f, dx, axis):
    return (np.roll(f, -1, axis=axis) - 2 * f + np.roll(f, 1, axis=axis)) / dx**2

# normalization
norm_ground = np.sqrt(np.sum(np.abs(psi_ground)**2) * dx * dy)
psi_ground /= norm_ground

# total ground state energy
#kinetic_ground = 0.5 * (np.abs(np.gradient(psi_ground, dx, axis=0))**2 + np.abs(np.gradient(psi_ground, dy, axis=1))**2)

kinetic_ground = -0.5*(np.conj(psi_ground) * second_derivative_nd(psi_ground, dx, axis=0) + np.conj(psi_ground) * second_derivative_nd(psi_ground, dy, axis=1))


potential_ground = V * np.abs(psi_ground)**2
energy_ground = np.sum(kinetic_ground + potential_ground) * dx * dy

#2D hydrogen first excited state：psi = (r - 3/4) * exp(-2/3 * r)
psi_excited = (r - 3/4) * np.exp(-2/3 * r)

# normalization fs
norm_excited = np.sqrt(np.sum(np.abs(psi_excited)**2) * dx * dy)
psi_excited /= norm_excited

# fs total energy
#kinetic_excited = 0.5 * (np.abs(np.gradient(psi_excited, dx, axis=0))**2 + np.abs(np.gradient(psi_excited, dy, axis=1))**2)

kinetic_excited = -0.5*(np.conj(psi_excited) * second_derivative_nd(psi_excited, dx, axis=0) + np.conj(psi_excited) * second_derivative_nd(psi_excited, dy, axis=1))

potential_excited = V * np.abs(psi_excited)**2
energy_excited = np.sum(kinetic_excited + potential_excited) * dx * dy

# Save wavefunctions to text files
ground_data = np.column_stack((X.flatten(), Y.flatten(), np.real(psi_ground).flatten(), np.imag(psi_ground).flatten()))
excited_data = np.column_stack((X.flatten(), Y.flatten(), np.real(psi_excited).flatten(), np.imag(psi_excited).flatten()))

np.savetxt("2d_ext_ground_state_wavefunction.txt", ground_data, header="x y Re(psi) Im(psi)", comments="")
np.savetxt("2d_ext_excited_state_wavefunction.txt", excited_data, header="x y Re(psi) Im(psi)", comments="")


#plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Ground state density
surfxy0 = ax[0].imshow(np.abs(psi_ground)**2, extent=(x.min(), x.max(), y.min(), y.max()), 
                       origin='lower', cmap='viridis')

ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_xlabel(r'$x$', rotation=0, fontsize=25)
ax[0].set_ylabel(r'$y$', rotation=0, fontsize=25)
ax[0].set_aspect('equal', adjustable='box')
ax[0].set_title(f'1s Density (Energy = {energy_ground:.8f})')
ax[0].set_xlim(shift, -shift)
ax[0].set_ylim(shift, -shift)
ax[0].text(0.1, 0.9, "(a)", fontsize=25, transform=ax[0].transAxes)

# ps script
ps.set_tick_inteval(ax[0].yaxis, major_itv=2.5, minor_itv=0.5)
ps.set_tick_inteval(ax[0].xaxis, major_itv=2.5, minor_itv=0.5)
ps.set(ax[0])
cbar0 = fig.colorbar(surfxy0, ax=ax[0])
cbar0.ax.tick_params(labelsize=25)

# Excited state density
surfxy1 = ax[1].imshow(np.abs(psi_excited)**2, extent=(x.min(), x.max(), y.min(), y.max()), 
                       origin='lower', cmap='plasma')

ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set_xlabel(r'$x$', rotation=0, fontsize=25)
ax[1].set_ylabel(r'$y$', rotation=0, fontsize=25)
ax[1].set_aspect('equal', adjustable='box')
ax[1].set_title(f'2s Density (Energy = {energy_excited:.8f})')
ax[1].set_xlim(shift, -shift)
ax[1].set_ylim(shift, -shift)
ax[1].text(0.1, 0.9, "(b)", fontsize=25, transform=ax[1].transAxes)


ps.set_tick_inteval(ax[1].yaxis, major_itv=2.5, minor_itv=0.5)
ps.set_tick_inteval(ax[1].xaxis, major_itv=2.5, minor_itv=0.5)

ps.set(ax[1])
cbar1 = fig.colorbar(surfxy1, ax=ax[1])
cbar1.ax.tick_params(labelsize=25)
plt.tight_layout()
plt.savefig(f"2d_ext_density_functions_{N}.pdf", format='pdf')
plt.show()

print("Wavefunctions saved as '2d_ext_ground_state_wavefunction.txt' and '2d_ext_excited_state_wavefunction.txt'.")
print("Density functions saved as '2d_ext_density_functions.pdf'.")

