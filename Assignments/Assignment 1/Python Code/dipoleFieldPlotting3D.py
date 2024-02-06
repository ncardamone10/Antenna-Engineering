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


# Define the meshgrid for the space
x = np.linspace(0.01*wavelength, 5 * wavelength, 10)  # 100 points from 0 to 50*wavelength
y = np.linspace(0.01*wavelength, 5 * wavelength, 10)
z = np.linspace(-5 * wavelength, 5 * wavelength, 20)
x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')

# Convert the grid to cylindrical coordinates
rho_grid = np.sqrt(x_grid**2 + y_grid**2)
phi_grid = np.arctan2(y_grid, x_grid)

# Compute the electric field components in cylindrical coordinates
Ez_grid = np.real(E_z(rho_grid, phi_grid, z_grid, I_0, k, z1, z2, z3))
Erho_grid = np.real(E_rho(rho_grid, phi_grid, z_grid, I_0, k, z1, z2, z3))
# Ephi is zero everywhere

# Convert cylindrical components to Cartesian for plotting
Ex = Erho_grid * np.cos(phi_grid) #- E_phi * np.sin(phi_grid)
Ey = Erho_grid * np.sin(phi_grid) #+ E_phi * np.cos(phi_grid)
Ez = Ez_grid  # No conversion needed for Ez

# Compute the magnitude of the electric field
E_magnitude = np.sqrt(np.abs(Ez)**2 + np.abs(Ey)**2 + np.abs(Ex)**2)

# Normalize the magnitude for coloring (in dB)
E_normalized = 20*np.log10(E_magnitude)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a colormap
norm = plt.Normalize(E_normalized.min(), E_normalized.max())
cmap = plt.get_cmap('jet')
colors = cmap(np.repeat(E_normalized, 3))


# Quiver plot
ax.quiver(x_grid/wavelength, y_grid/wavelength, z_grid/wavelength, Ex, Ey, Ez, color=colors, length=3*wavelength,  normalize=True)
#length=wavelength, normalize=True

# Create a colorbar
mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(E_normalized)
plt.colorbar(mappable, ax=ax, label='Electric Field Magnitude (dBV/m)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Electric Field Vector Field')

plt.show()

# Just plotting the field magnitude
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Scatter plot
ax.scatter(x_grid, y_grid, z_grid, c=E_magnitude, cmap='jet', marker='o')

# Create a colorbar
mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(E_normalized)
cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Electric Field Magnitude (dB)')

ax.set_xlabel('X (λ)')
ax.set_ylabel('Y (λ)')
ax.set_zlabel('Z (λ)')
ax.set_title('3D Scatter Plot of Electric Field Magnitude')

plt.show()