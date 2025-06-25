import numpy as np
import matplotlib.pyplot as plt

# Create 3D meshgrid
N = 8
shift = -6

x = np.linspace(shift, -shift, 2**N)
y = np.linspace(shift, -shift, 2**N)
z = np.linspace(shift, -shift, 2**N)
X, Y, Z = np.meshgrid(x, y, z)

# Define spherical radius
r = np.sqrt(X**2 + Y**2 + Z**2)

# Define wavefunctions
psi_ground = np.exp(-r)
psi_excited = (r - 2) * np.exp(-0.5 * r)

# Step size
dx = x[1] - x[0]

# Normalize wavefunctions
norm_ground = np.sqrt(np.sum(np.abs(psi_ground)**2) * dx**3)
norm_excited = np.sqrt(np.sum(np.abs(psi_excited)**2) * dx**3)
psi_ground /= norm_ground
psi_excited /= norm_excited

# Compute energies
V = -1 / r
kinetic_ground = 0.5 * (np.sum(np.gradient(psi_ground, dx, axis=0)**2) +
                        np.sum(np.gradient(psi_ground, dx, axis=1)**2) +
                        np.sum(np.gradient(psi_ground, dx, axis=2)**2)) * dx**3
potential_ground = np.sum(V * np.abs(psi_ground)**2) * dx**3
energy_ground = kinetic_ground + potential_ground

kinetic_excited = 0.5 * (np.sum(np.gradient(psi_excited, dx, axis=0)**2) +
                         np.sum(np.gradient(psi_excited, dx, axis=1)**2) +
                         np.sum(np.gradient(psi_excited, dx, axis=2)**2)) * dx**3
potential_excited = np.sum(V * np.abs(psi_excited)**2) * dx**3
energy_excited = kinetic_excited + potential_excited

# Project densities onto planes
Z0 = psi_ground
Z1 = psi_excited
INTZ_Z0 = np.sum(np.abs(Z0)**2, axis=2) * dx
INTZ_Z1 = np.sum(np.abs(Z1)**2, axis=2) * dx
INTX_Z0 = np.sum(np.abs(Z0)**2, axis=0) * dx
INTX_Z1 = np.sum(np.abs(Z1)**2, axis=0) * dx
INTY_Z0 = np.sum(np.abs(Z0)**2, axis=1) * dx
INTY_Z1 = np.sum(np.abs(Z1)**2, axis=1) * dx

# Plot densities
plt.figure(figsize=(12, 24))

# XY plane
plt.subplot(3, 2, 1)
plt.imshow(INTZ_Z0, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.title(f'1s XY Density (Energy = {energy_ground:.8f})')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(3, 2, 2)
plt.imshow(INTZ_Z1, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='plasma')
plt.colorbar(label='Intensity')
plt.title(f'2s XY Density (Energy = {energy_excited:.8f})')
plt.xlabel('x')
plt.ylabel('y')

# XZ plane
plt.subplot(3, 2, 3)
plt.imshow(INTY_Z0, extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.title(f'1s XZ Density (Energy = {energy_ground:.8f})')
plt.xlabel('x')
plt.ylabel('z')

plt.subplot(3, 2, 4)
plt.imshow(INTY_Z1, extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='plasma')
plt.colorbar(label='Intensity')
plt.title(f'2s XZ Density (Energy = {energy_excited:.8f})')
plt.xlabel('x')
plt.ylabel('z')

# YZ plane
plt.subplot(3, 2, 5)
plt.imshow(INTX_Z0, extent=(y.min(), y.max(), z.min(), z.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.title(f'1s YZ Density (Energy = {energy_ground:.8f})')
plt.xlabel('y')
plt.ylabel('z')

plt.subplot(3, 2, 6)
plt.imshow(INTX_Z1, extent=(y.min(), y.max(), z.min(), z.max()), origin='lower', cmap='plasma')
plt.colorbar(label='Intensity')
plt.title(f'2s YZ Density (Energy = {energy_excited:.8f})')
plt.xlabel('y')
plt.ylabel('z')

plt.tight_layout()
plt.savefig("3d_ext_density_functions.pdf", format='pdf')
plt.show()

print("Wavefunctions and densities plotted and saved.")

