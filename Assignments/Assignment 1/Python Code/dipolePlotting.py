import numpy as np
import matplotlib.pyplot as plt
from ai import cs
from mpl_toolkits.mplot3d import Axes3D

# Constants (replace with actual values)
I_0 = 1
f = 2e9 # 2 GHz Dipole
c = 3e8 # speed of light
wavelength = c / f
k = 2 * np.pi / wavelength  # Assuming a wavelength of 1 for simplicity
h = 0.25*wavelength  # Assuming the total length of the antenna is 1
z1 = -h
z2 = 0
z3 = h
eta = 120*np.pi


# Define the range of z values
z = np.linspace(z1, z3, 1000)

# Define the current distribution function I(z)
def I(z, I_0, k, h, z1, z2, z3):
    return np.piecewise(z, [z > z2, z <= z2],
                        [lambda z: I_0 * np.sin(k * (z3 - z)) / np.sin(k * (z3 - z2)),
                         lambda z: I_0 * np.sin(k * (z - z1)) / np.sin(k * (z2 - z1))])
# Calculate the current values using the piecewise function
current = I(z, I_0, k, h, z1, z2, z3)


# Functions for R1, R2, and R3
def R1(x, y, z, z1):
    return np.sqrt(x**2 + y**2 + (z1 - z)**2)

def R2(x, y, z, z2):
    return np.sqrt(x**2 + y**2 + (z2 - z)**2)

def R3(x, y, z, z3):
    return np.sqrt(x**2 + y**2 + (z3 - z)**2)

# Function for rho
def rho(x, y):
    return np.sqrt(x**2 + y**2)

# Function for the E-field components
def E_z(x, y, z, I_0, k, z1, z2, z3):
    rho_val = rho(x, y)
    R1_val = R1(x, y, z, z1)
    R2_val = R2(x, y, z, z2)
    R3_val = R3(x, y, z, z3)
    h1 = z2 - z1
    h2 = z3 - z2
    temp = (1/np.tan(k * h1) + 1/np.tan(k * h2))
    temp = temp * np.exp(-1j * k * R2_val) / R2_val
    temp = temp - np.exp(-1j * k * R1_val) / (R1_val * np.sin(k * h1))
    temp = temp - np.exp(-1j * k * R3_val) / (R3_val * np.sin(k * h2))
    return 30j*I_0*temp

def E_rho(x, y, z, I_0, k, z1, z2, z3):
    rho_val = rho(x, y)
    R1_val = R1(x, y, z, z1)
    R2_val = R2(x, y, z, z2)
    R3_val = R3(x, y, z, z3)
    h1 = z2 - z1
    h2 = z3 - z2
    temp = (z - z1) * np.exp(-1j * k * R1_val) / (R1_val * np.sin(k * h1))
    temp = temp - (z - z2) * np.exp(-1j * k * R2_val) / R2_val * (1/np.tan(k * h1) + 1/np.tan(k * h2))
    temp = temp + (z - z3) * np.exp(-1j * k * R3_val) / (R3_val * np.sin(k * h2))
    return 30j*I_0*temp/rho_val


def E_phi(x, y, z, I_0, k, z1, z2, z3):
    return 0*(z + x + y)


# Plotting the E fields along r with plane cuts
# Define the range of r, theta, phi values
r = np.logspace(-3, 4, 1000)
theta = 0*np.ones(np.size(r))
phi = 0*np.ones(np.size(r))

# Convert to rect coords
xE, yE, zE = cs.sp2cart(r, theta, phi)

# Generate E Fields
Ex, Ey, Ez = cs.cyl2cart(E_rho(xE, yE, zE, I_0, k, z1, z2, z3), E_phi(xE, yE, zE, I_0, k, z1, z2, z3), E_z(xE, yE, zE, I_0, k, z1, z2, z3))
# print(f'Ex type: {type(Ex)}')
# print(f'Ey type: {type(Ey)}')
# print(f'Ez type: {type(Ez)}')
# print(f'Ex shape: {np.ndim(Ex)}')
# print(f'Ey shape: {np.ndim(Ey)}')
# print(f'Ez shape: {np.ndim(Ez)}')
# Er, Etheta, Ephi = cs.cart2sp(Ex, Ey, Ez)

Er = np.zeros_like(r)
Etheta = np.zeros_like(r)
Ephi = np.zeros_like(r)

def cart2sp(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)  # Radius
    theta = np.arccos(z / r)  # Inclination
    phi = np.arctan(y/(x+1e-10*np.ones_like(x)))  # Azimuth
    return r, theta, phi

for i in range(len(r)):
    Er[i], Etheta[i], Ephi[i] = cart2sp(Ex[i], Ey[i], Ez[i])


# # Plot the current distribution
# plt.figure(figsize=(10, 6))
# plt.plot(z/h, current/I_0, label='Normalized Current Distribution $I(z)/I_0$')
# plt.axvline(x=z1/h, color='r', linestyle='--', label='$z_1$')
# plt.axvline(x=z2/h, color='g', linestyle='--', label='$z_2$')
# plt.axvline(x=z3/h, color='b', linestyle='--', label='$z_3$')
# plt.xlabel('z/h')
# plt.ylabel('$I(z)/I_0$')
# plt.title('Normalized Current Distribution of a Dipole Antenna')
# plt.legend()
# plt.grid(True)
# plt.show()

# Plotting the results in a 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Current distribution plot
axs[0, 0].plot(z/h, current/I_0, label='Normalized Current Distribution $I(z)/I_0$')
axs[0, 0].set_xlabel('z/h')
axs[0, 0].set_ylabel('$I(z)/I_0$')
axs[0, 0].set_title('Normalized Current Distribution of a Dipole Antenna')
axs[0, 0].legend()
axs[0, 0].grid(True)

# E_r magnitude plot in dB
axs[0, 1].plot(r, 20*np.log10(np.abs(Er)), label='$E_r$ in dB')
axs[0, 1].set_xlabel('r')
axs[0, 1].set_ylabel('$E_r$ (dB)')
axs[0, 1].set_title('$E_r$ Magnitude in dB')
axs[0, 1].legend()
axs[0, 1].grid(True)

# E_theta magnitude plot in dB
axs[1, 0].plot(r, 20*np.log10(np.abs(Etheta)), label='$E_\\theta$ in dB')
axs[1, 0].set_xlabel('r')
axs[1, 0].set_ylabel('$E_\\theta$ (dB)')
axs[1, 0].set_title('$E_\\theta$ Magnitude in dB')
axs[1, 0].legend()
axs[1, 0].grid(True)

# E_phi magnitude plot in dB
axs[1, 1].plot(r, 20*np.log10(np.abs(Ephi)), label='$E_\\phi$ in dB')
axs[1, 1].set_xlabel('r')
axs[1, 1].set_ylabel('$E_\\phi$ (dB)')
axs[1, 1].set_title('$E_\\phi$ Magnitude in dB')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()
