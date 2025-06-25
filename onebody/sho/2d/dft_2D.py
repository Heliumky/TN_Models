import numpy as np
import matplotlib.pyplot as plt

# meshgrid
x = np.linspace(-4, 4, 64)
y = np.linspace(-4, 4, 64)
X, Y = np.meshgrid(x, y)

# SHO and 2D Hydrogen
V = -1 / np.sqrt(X**2 + Y**2)  
#V = 0.5*(X**2 + Y**2)

# initial wavefunc
np.random.seed(42)
real_part = np.random.rand(*X.shape)
imag_part = np.random.rand(*X.shape)
psi = (real_part + 1j * imag_part).astype(np.complex128)
# init gaussian func
#psi = np.exp(-(X**2 + Y**2)).astype(np.complex128)

# normalization
dx = x[1] - x[0]
dy = y[1] - y[0]
norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
psi /= norm

# build up k space
kx = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
ky = 2 * np.pi * np.fft.fftfreq(len(y), d=dy)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

# Imaginary time evolution parameters
tau = 0.01
max_steps_gd = 10000
max_steps_fs = 40000

# grond state
psi_k = np.fft.fft2(psi) * np.exp(K2 / 4 * tau)
for step in range(max_steps_gd):
    psi = np.fft.ifft2(psi_k)
    psi *= np.exp(-V * tau)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    psi /= norm
    psi_k = np.fft.fft2(psi)
    psi_k *= np.exp(-K2 * tau / 2)

psi_k *= np.exp(K2 * tau / 4)
psi_ground = np.fft.ifft2(psi_k)
psi_ground /= np.sqrt(np.sum(np.abs(psi_ground)**2) * dx * dy)

# grond state energy
energy_ground = np.sum(
    0.5 * (np.abs(np.gradient(psi_ground, dx, axis=0))**2 + np.abs(np.gradient(psi_ground, dy, axis=1))**2)
    + V * np.abs(psi_ground)**2
) * dx * dy

# init first-excited state
np.random.seed(43)
real_part = np.random.rand(*X.shape) - 0.5
imag_part = np.random.rand(*X.shape) - 0.5
psi_excited = (real_part + 1j * imag_part).astype(np.complex128)

# init gaussian func
#psi = np.exp(-(X**2 + Y**2)).astype(np.complex128)

norm = np.sqrt(np.sum(np.abs(psi_excited)**2) * dx * dy)
psi_excited /= norm

# Imaginary time evolution parameters
psi_k = np.fft.fft2(psi_excited) * np.exp(K2 / 4 * tau)
for step in range(max_steps_fs):
    psi_excited = np.fft.ifft2(psi_k)
    psi_excited *= np.exp(-V * tau)

    # make the excited state orthogonal with ground state
    overlap = np.sum(np.conj(psi_ground) * psi_excited) * dx * dy
    psi_excited -= overlap * psi_ground

    # normalization
    norm = np.sqrt(np.sum(np.abs(psi_excited)**2) * dx * dy)
    psi_excited /= norm
    psi_k = np.fft.fft2(psi_excited)
    psi_k *= np.exp(-K2 * tau / 2)

psi_k *= np.exp(K2 * tau / 4)
psi_excited = np.fft.ifft2(psi_k)
psi_excited /= np.sqrt(np.sum(np.abs(psi_excited)**2) * dx * dy)

# fs energy
energy_excited = np.sum(
    0.5 * (np.abs(np.gradient(psi_excited, dx, axis=0))**2 + np.abs(np.gradient(psi_excited, dy, axis=1))**2)
    + V * np.abs(psi_excited)**2
) * dx * dy


# Save wavefunctions to text files
ground_data = np.column_stack((X.flatten(), Y.flatten(), np.real(psi_ground).flatten(), np.imag(psi_ground).flatten()))
excited_data = np.column_stack((X.flatten(), Y.flatten(), np.real(psi_excited).flatten(), np.imag(psi_excited).flatten()))

np.savetxt("2d_dftimt_ground_state_wavefunction.txt", ground_data, header="x y Re(psi) Im(psi)", comments="")
np.savetxt("2d_dftimt_excited_state_wavefunction.txt", excited_data, header="x y Re(psi) Im(psi)", comments="")

# Save density plots as PDF
plt.figure(figsize=(12, 6))

# Ground state density
plt.subplot(1, 2, 1)
plt.imshow(np.abs(psi_ground)**2, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.title(f'Ground State Density (Energy = {energy_ground:.8f})')
plt.xlabel('x')
plt.ylabel('y')

# Excited state density
plt.subplot(1, 2, 2)
plt.imshow(np.abs(psi_excited)**2, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='plasma')
plt.colorbar(label='Intensity')
plt.title(f'First Excited State Density (Energy = {energy_excited:.8f})')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.savefig("2d_dftimt_density_functions.pdf", format='pdf')
plt.show()

print("Wavefunctions saved as '2d_dftimt_ground_state_wavefunction.txt' and '2d_dftimt_excited_state_wavefunction.txt'.")
print("Density functions saved as '2d_dftimt_density_functions.pdf'.")

