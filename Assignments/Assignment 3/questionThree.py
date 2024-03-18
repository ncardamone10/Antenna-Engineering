'''
This solves question 3 of the antennas assignment 3
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy.integrate import simpson
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# General Params
f = 1e9
c = 3e8
k = 2*np.pi*f/c
wavelength = c/f

EPSILON = 1e-16


# Define tau limits of integration
def getTau1(beta, h, k, theta):
    return -np.sqrt(2 * beta / np.pi) * ( h + k * np.cos(theta) / ( 2 * beta + EPSILON) )

def getTau2(beta, h, k, theta):
    return np.sqrt(2 * beta / np.pi) * ( h - k * np.cos(theta) / ( 2 * beta + EPSILON) )

# Define f_z from radiation integrals
def getFz(beta, k, theta, tau1, tau2, I0=1):
    S1, C1 = sc.fresnel(tau1)
    S2, C2 = sc.fresnel(tau2)

    term1 = np.sqrt(np.pi / (2 * beta + EPSILON))*I0
    term2 = np.exp(-1j * (k * theta)**2 / ( 4 * beta+ EPSILON))

    result = term1 * term2 * (C2 - C1 + 1j * (S2 - S1)) 
    return result

# Define E field (analytically from radiation integrals)
def getE(beta, h, k, theta, I0=1, eta=120*np.pi):
    tau1 = getTau1(beta, h, k, theta)
    tau2 = getTau2(beta, h, k, theta)
    fz = getFz(beta, k, theta, tau1, tau2, I0)
    return fz * np.sin(theta) * 1j * k * eta / (4 * np.pi)

# Define Radiation Intensity (based on E field)
def getU(beta, h, k, theta, I0=1, eta=120*np.pi):
    E = getE(beta, h, k, theta, I0, eta)
    return np.abs(E)**2 / (2 * eta)

# Define Directivity  (based on U and weighted average)
def getD(beta, h, k, theta, I0=1, eta=120*np.pi):
    U = getU(beta, h, k, theta, I0, eta)
    sinTheta = np.sin(theta)
    Uavg = np.average(U, weights=sinTheta)
    return U / Uavg

# Antenna Dimensions to sweep over
h = 5 * wavelength
Phi0MaxScaler = 50
Phi0 = np.linspace(0, Phi0MaxScaler / h**2, 50)
beta = Phi0 / h**2

# Plotting Stuff
theta = np.linspace(0.1, np.pi - 0.1, 1000)

# Plot 3D surface of U over Phi0 and Theta
def plot_3D_surface_U(k, h, theta, Phi0, I0=1, eta=120*np.pi):
    # Convert theta and Phi0 into 2D grid arrays
    Theta, Phi0_grid = np.meshgrid(theta, Phi0)
    beta_grid = Phi0_grid / h**2
    
    # Initialize an array to hold U values
    U_dBi_grid = np.zeros_like(Theta)
    
    # Compute U for each combination of theta and Phi0
    for i in range(len(Phi0)):
        for j in range(len(theta)):
            U = getU(beta_grid[i, j], h, k, Theta[i, j], I0, eta)
            U_dBi = 10 * np.log10(U)  # Convert U to dBi
            U_dBi_grid[i, j] = U_dBi
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(Theta * 180 / np.pi, Phi0_grid, U_dBi_grid, cmap='jet')
    
    # Labels and title
    ax.set_xlabel('Theta (Degrees)')
    ax.set_ylabel('Phi0')
    ax.set_zlabel('U (dBi)')
    ax.set_title('Radiation Intensity U in dBi')
    ax.set_zlim(-30, 30)
    # Colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5, label='U (dBi)')
    
    plt.show()

# Plot 3D surface of D over Phi0 and Theta
def plot_3D_directivity(beta, h, k, theta, I0=1, eta=120*np.pi):
    Theta, Beta = np.meshgrid(theta, beta)  # Create a meshgrid for theta and beta
    
    # Initialize directivity array for all beta and theta combinations
    D_dBi = np.zeros_like(Theta)
    
    # Compute directivity for each beta across all theta at once
    for i, beta_value in enumerate(beta):
        D = getD(beta_value, h, k, theta, I0, eta)
        D_dBi[i, :] = 10 * np.log10(D)  # Convert to dBi for plotting
    
    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Theta * 180 / np.pi, Beta, D_dBi, cmap='jet')  # Theta converted to degrees

    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Phi0')
    ax.set_zlabel('Directivity (dBi)')
    ax.set_title('3D Surface Plot of Directivity')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()

# Plot the Maximum Directivity vs. Phi0
def plot_max_directivity_vs_phi0(beta, h, k, theta, I0=1, eta=120*np.pi):
    # Initialize array to store the maximum directivity for each beta
    max_dBi = []

    # Loop over each beta (related to Phi0) value
    for b in beta:
        # Calculate directivity for all theta at once
        D = getD(b, h, k, theta, I0, eta)
        D_dBi = 10 * np.log10(D)  # Convert to dBi

        # Find the maximum directivity for this beta
        max_dBi.append(np.max(D_dBi))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(beta * h**2, max_dBi, label='Max Directivity')
    plt.xlabel('$\Phi_0$')
    plt.ylabel('Maximum Directivity (dBi)')
    plt.title('Maximum Directivity vs. $\Phi_0$')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot the Maximum Directivity vs. Phi0 and h/lambda
def plot_3D_max_directivity_vs_phi0_h(beta_range, h_over_lambda_range, k, theta, I0=1, eta=120*np.pi):
    # Preallocate array for maximum directivity values
    max_dBi_grid = np.zeros((len(beta_range), len(h_over_lambda_range)))

    # Generate theta values in radians
    theta = np.linspace(0, np.pi, 1000)
    
    # Loop over each beta and h/wavelength value
    for i, Phi0 in enumerate(beta_range):
        for j, h_over_lambda in enumerate(h_over_lambda_range):
            h = h_over_lambda * wavelength
            beta = Phi0 / h**2
            
            # Calculate directivity for all theta at once
            D = getD(beta, h, k, theta, I0, eta)
            D_dBi = 10 * np.log10(D)  # Convert to dBi

            # Find the maximum directivity for this combination
            max_dBi_grid[i, j] = np.max(D_dBi)

    # Plotting
    Phi0_grid, h_over_lambda_grid = np.meshgrid(beta_range, h_over_lambda_range)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(Phi0_grid, h_over_lambda_grid, max_dBi_grid.T, cmap='jet')
    ax.set_xlabel('$\Phi_0$')
    ax.set_ylabel('$h/\lambda$')
    ax.set_zlabel('Max Directivity (dBi)')
    ax.set_title('Max Directivity vs. $\Phi_0$ and $h/\lambda$')
    fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar
    
    plt.show()



plot_3D_surface_U(k, h, theta, Phi0)
plot_3D_directivity(beta, h, k, theta)
plot_max_directivity_vs_phi0(beta, h, k, theta)    


# Parameters setup
Phi0MaxScaler = 10
beta_range = np.linspace(0, Phi0MaxScaler / (1 * wavelength)**2, 500)  # Phi0 range
h_over_lambda_range = np.linspace(1, 20, 100)  # h/wavelength range

# Example call to the plotting function
plot_3D_max_directivity_vs_phi0_h(beta_range, h_over_lambda_range, k, theta)