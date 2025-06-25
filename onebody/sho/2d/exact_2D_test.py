import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, eval_genlaguerre


# Define parameters

n = 1   # Principal quantum number
m = 0   # Magnetic quantum number
#E_q0_relation = -q0**2
E_n_relation = -1 / (2*((n + 0.5)**2))
q0 =np.sqrt(-E_n_relation*2)


def psi_nm(rho, phi, n, m, q0):
    norm_factor = np.sqrt(q0**3 * factorial(n - abs(m)) / (np.pi * factorial(n + abs(m))))
    radial_part = (2 * q0 * rho) ** abs(m) * np.exp(-q0 * rho) * eval_genlaguerre(n - abs(m), 2 * abs(m), 2 * q0 * rho)
    angular_part = np.exp(1j * m * phi)
    return norm_factor*radial_part * angular_part


# Define density function
def density(rho, phi, n, m, q0):
    return np.abs(psi_nm(rho, phi, n, m, q0))**2

# Grid for plotting
rho_vals = np.linspace(0, 5, 100)
phi_vals = np.linspace(0, 2 * np.pi, 100)
rho, phi = np.meshgrid(rho_vals, phi_vals)

# Convert to Cartesian coordinates
x = rho * np.cos(phi)
y = rho * np.sin(phi)

# Compute density
density_vals = density(rho, phi, n, m, q0)

# Plot
plt.figure(figsize=(7, 6))
plt.pcolormesh(x, y, density_vals, shading='auto', cmap='plasma')
plt.colorbar(label=r'$|\Psi_{nm}(\rho, \phi)|^2$')
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Density Plot for n={n}, m={m}\nE = {E_n_relation:.3f}, E = {E_n_relation:.3f}")
plt.show()
