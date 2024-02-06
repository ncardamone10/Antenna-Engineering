import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# Constants (replace with actual values)
I_0 = 1
f = 2e9 # 2 GHz Dipole
c = 3e8 # speed of light
# wavelength = c / f
wavelength = 100
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
    # x = np.abs(x)
    # y = np.abs(y)
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
numberOfPoints = 500
# Spherical Coords to Sweep Over
theta = np.linspace(0, np.pi-0.001, numberOfPoints)
phi = np.zeros_like(theta)
thetaNormFactor = np.pi/2
phiNormFactor = 0

# Desired r values
r_values = [0.1 * wavelength, 0.2 * wavelength, 0.3*wavelength, 0.4 * wavelength, 0.5 * wavelength, 1 * wavelength, 2 * wavelength, 5 * wavelength, 
            10 * wavelength, 20 * wavelength, 50 * wavelength]
# Prepare the plots correctly outside the loop
fig = plt.figure(figsize=(12, 14))
gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.5)  # Adjust hspace for spacing

ax_rect = fig.add_subplot(gs[0])
ax_polar = fig.add_subplot(gs[1], polar=True)

# Variables to collect labels and handles for the legend
labels = []
handles = []

for r in r_values:
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

    # Loop through all points in space and calculate the E field (sweep through theta)
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

    # Calculate Etheata normalization factor
    # Matrix to convert from cylindrical coord to rect (need to inverse it though)
    rectToCyl = np.array([
        [np.cos(phiNormFactor),  np.sin(phiNormFactor), 0],
        [-np.sin(phiNormFactor), np.cos(phiNormFactor), 0],
        [0,                      0,                     1]
    ])
    # Calculate EnormFactor
    xN = r * np.sin(thetaNormFactor) * np.cos(phiNormFactor)
    yN = r * np.sin(thetaNormFactor) * np.sin(phiNormFactor)
    zN = r * np.cos(thetaNormFactor)

    # Generate E fields (cylindrical coords)
    ErhoNormFactor = E_rho(xN,yN,zN,I_0,k,z1,z2,z3)
    EphiNormFactor = E_phi(xN,yN,zN,I_0,k,z1,z2,z3)
    EzNormFactor   = E_z(xN,yN,zN,I_0,k,z1,z2,z3)

    # Column vector of the E field at one point in space (cylindrical coords)
    Ecyl = np.array([
        [ErhoNormFactor],
        [EphiNormFactor],
        [EzNormFactor]
    ])
    # Column vector of the E field at one point in space (rect coords)
    Ecart = np.linalg.inv(rectToCyl) @ Ecyl
    # Matrix to convert from rect to spherical
    rectToSph = np.array([
        [np.sin(thetaNormFactor)*np.cos(phiNormFactor), np.sin(thetaNormFactor)*np.sin(phiNormFactor),  np.cos(thetaNormFactor)],
        [np.cos(thetaNormFactor)*np.cos(phiNormFactor), np.cos(thetaNormFactor)*np.sin(phiNormFactor), -np.sin(thetaNormFactor)],
        [-np.sin(phiNormFactor),                        np.cos(phiNormFactor),                          0                      ]
    ])
    # Column vector of the E field at one point in space (spherical coords)
    Esph = rectToSph @ Ecart
    # Save E field 
    EthetaNormFactor = Esph[1]

    # Convert to dBV/m
    Er_dB = 20 * np.log10(np.abs(Er))
    Etheta_dB = 20 * np.log10(np.abs(Etheta))
    EthetaNormFactor_dB = 20 * np.log10(np.abs(EthetaNormFactor))

    # Normalize Etheta_dB as needed before plotting
    Etheta_dB_normalized = Etheta_dB - EthetaNormFactor_dB  # Example normalization step

    label = f'r = {r/wavelength:.3f}λ'
    

    # Add in the negative theta part
    thetaSym = np.concatenate((-np.flip(theta), theta))
    EthetaSym = np.concatenate((np.flip(Etheta_dB_normalized), Etheta_dB_normalized))

    # Plot on the rectangular axis
    line, = ax_rect.plot(np.degrees(thetaSym), EthetaSym, label=label)
    # Plot on the polar axis, but no need to collect handles again
    ax_polar.plot(thetaSym, EthetaSym, label=label)
    
    labels.append(label)
    handles.append(line)

# Configure the rectangular plot
ax_rect.set_xlabel(r'$\theta$ (degrees)')
ax_rect.set_ylabel(r'$E_{\theta}$ (dBu)')
ax_rect.set_title(f'Rectangular Plot of Normalized $|E_{{\\theta}}|$ vs $\\theta$ for $h = {h/wavelength:.3f}\\lambda$ and sweeping $r$')
ax_rect.grid(True)

ax_rect.set_xticks(np.arange(-180, 181, 45))  # Adjust ticks to remove unwanted labels

# Configure the polar plot
ax_polar.set_theta_zero_location('N')  # 0 degrees at the top
ax_polar.set_theta_direction(-1)  # Clockwise
ax_polar.set_title(f'Polar Plot of Normalized $|E_{{\\theta}}|$ vs $\\theta$ for $h = {h/wavelength:.3f}\\lambda$ and sweeping $r$')
ax_polar.grid(True)
ax_polar.set_ylim(-40, 0)

# Create a shared legend
# Adjusting layout for better control over legend placement
plt.tight_layout()

# Adding an external legend below the plots
# The 'bbox_to_anchor' coordinates might need adjustment depending on your figure size and layout
fig.subplots_adjust(bottom=0.1)
legend = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.8, 0.35), ncol=1)

plt.show()