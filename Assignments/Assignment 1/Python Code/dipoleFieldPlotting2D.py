import numpy as np
import matplotlib.pyplot as plt
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

# Goal: plot magnitude of Etheta and Er (dB) for r in range(0,50*wavelength), when theta = 90 degrees, phi=0
numberOfPoints = 10000
# Spherical Coords to Sweep Over
r = np.linspace(0.01*wavelength, 200*wavelength,numberOfPoints)
phi = np.zeros_like(r)
theta = np.pi/2*np.ones_like(r)

# Convert to Cylindrical Coords
# rho = r*np.sin(theta)
# z = r*np.cos(theta)
# phi remains unchanged 

# Convert to Rect Coords
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# Generate E fields (cylindrical coords)
Erho = E_rho(x,y,z,I_0,k,z1,z2,z3)
Ephi = E_phi(x,y,z,I_0,k,z1,z2,z3)
Ez   = E_z(x,y,z,I_0,k,z1,z2,z3)


# Convert from Cylindrical to Spherical Coords for E field
Er = np.zeros_like(Erho)
Etheta = np.zeros_like(Erho)
# Ephi should be unchanged
for i in range(0,numberOfPoints):
    # Matrix to convert from cylindrical coord to rect (need to inverse it though)
    rectToCyl = np.array([
        [np.cos(phi[i]),  np.sin(phi[i]), 0],
        [-np.sin(phi[i]), np.cos(phi[i]), 0],
        [0,               0,              1]
    ])
    # Column vector of the E field at one point in space (cylindrical coords)
    Ecyl = np.array([
        [Erho[i]],
        [Ephi[i]],
        [Ez[i]]
    ])
    # Column vector of the E field at one point in space (rect coords)
    Ecart = np.linalg.inv(rectToCyl) @ Ecyl
    # Matrix to convert from rect to spherical
    rectToSph = np.array([
        [np.sin(theta[i])*np.cos(phi[i]), np.sin(theta[i])*np.sin(phi[i]),  np.cos(theta[i])],
        [np.cos(theta[i])*np.cos(phi[i]), np.cos(theta[i])*np.sin(phi[i]), -np.sin(theta[i])],
        [-np.sin(phi[i]),                 np.cos(phi[i]),                   0               ]
    ])
    # Column vector of the E field at one point in space (spherical coords)
    Esph = rectToSph @ Ecart
    # Save E field to array
    Er[i] = (Esph[0])
    Etheta[i] = (Esph[1])

    # Convert magnitudes to dB
Er_dB = 20 * np.log10(np.abs(Er))
Etheta_dB = 20 * np.log10(np.abs(Etheta))

# Creating the 2x1 plot
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot Er magnitude in dB
axs[0].plot(r / wavelength, Er_dB, label='$E_r$ Magnitude')
axs[0].set_xlabel('$r/\lambda$')
axs[0].set_ylabel('$|E_r|$ (dB)', rotation='horizontal', labelpad=10, ha='right')
axs[0].set_title('$E_r$ Magnitude vs $r$')
axs[0].grid(True)
axs[0].legend()

# Plot Etheta magnitude in dB
axs[1].plot(r / wavelength, Etheta_dB, label='$E_\\theta$ Magnitude')
axs[1].set_xlabel('$r/\lambda$')
axs[1].set_ylabel('$|E_\\theta|$ (dB)', rotation='horizontal', labelpad=10, ha='right')
axs[1].set_title('$E_\\theta$ Magnitude vs $r$')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
