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
import matplotlib.cm as cm
import matplotlib.ticker as ticker

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
    term2 = np.exp(-1j * (k * np.cos(theta))**2 / ( 4 * beta+ EPSILON))

    result = term1 * term2 * (C2 - C1 - 1j * (S2 - S1)) 
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

# Plotting Functions

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
    surf = ax.plot_surface(Theta * 180 / np.pi, Phi0_grid/np.pi, U_dBi_grid - np.max(U_dBi_grid), cmap='jet')
    
    
    # Labels and title
    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.arange(0, 180.1, 22.5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.1f}$^\degree$'.format(val)))
    
    ax.set_ylabel('$\Phi_0$')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.0f}$^\degree$'.format(val)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.1f}$\pi$'.format(val)))
    
    ax.set_zlabel('U (dBr)')
    ax.set_title('Normalized Radiation Intensity While Adding Quadratic Phase Error')
    #ax.set_zlim(-30, 0)
    # Colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5, label='U (dBr)')
    
    # Add mesh lines
    # Use every 10th point in the x and y directions
    pointSkip = 1
    ax.plot_wireframe(Theta[::pointSkip, ::pointSkip] * 180 / np.pi, Phi0_grid[::pointSkip, ::pointSkip]/np.pi, U_dBi_grid[::pointSkip, ::pointSkip] - np.max(U_dBi_grid), color='k', linewidth=0.6)
    
    
    plt.show()
# Plot 2D curves of U over Theta for different Phi0
def plot_2D_curves_U(k, h, theta, Phi0, I0=1, eta=120*np.pi):
    # Convert theta and Phi0 into 2D grid arrays
    # Theta, Phi0_grid = np.meshgrid(theta, Phi0)
    # beta_grid = Phi0_grid / h**2
    beta = Phi0 / h**2
    
    # Create a color map with reversed colors
    colours = cm.rainbow(np.linspace(1, 0, len(Phi0)))
    
    # Compute U for each combination of theta and Phi0 and plot the results
    maxU = 0
    for beta0,colour in zip(beta, colours):
        U = getU(beta0, h, k, theta, I0, eta)
        U_dBi = 10 * np.log10(U)  # Convert U to dBi
        U_dBi = np.array(U_dBi)
        U_dBi = np.array(U_dBi)  
        if np.max(U_dBi) > maxU:
            maxU = np.max(U_dBi)          
        plt.plot(theta * 180 / np.pi, U_dBi - maxU, color=colour, label=f'$\Phi_0$={beta0/np.pi*h**2:.2f}$\pi$', linewidth=2)
    
    # Labels, title, and legend
    plt.xlabel(f'$\\theta$')
    
    plt.ylabel('U (dBr)')
    plt.title('Normalized Radiation Intensity While Adding Quadratic Phase Error (2D For Derek)')
    plt.legend()
    
    # Grid
    plt.grid(True)
    
    # Set x-axis ticks and labels
    plt.xticks(np.arange(0, 180.1, 22.5))
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f$\degree$'))
    
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
    surf = ax.plot_surface(Theta * 180 / np.pi, Beta*h**2/np.pi, D_dBi, cmap='jet')  # Theta converted to degrees

    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.arange(0, 180.1, 22.5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.1f}$^\degree$'.format(val)))
    
    ax.set_ylabel('$\\Phi_0$')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.0f}$^\degree$'.format(val)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.1f}$\pi$'.format(val)))
    
    ax.set_zlabel('Directivity (dBi)')
    ax.set_title('3D Surface Plot of Directivity (While Adding Quadratic Phase Error)')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Add mesh lines
    # Use every 10th point in the x and y directions
    pointSkip = 1
    ax.plot_wireframe(Theta[::pointSkip, ::pointSkip] * 180 / np.pi, Beta[::pointSkip, ::pointSkip]*h**2/np.pi, D_dBi[::pointSkip, ::pointSkip], color='k', linewidth=0.6)
    

    plt.show()
# Plot 2D curves of D over Theta for different Beta
def plot_2D_curves_D(k, h, theta, beta, I0=1, eta=120*np.pi):
    # Create a color map with reversed colors
    colors = cm.rainbow(np.linspace(1, 0, len(beta)))
    
    # Compute D for each combination of theta and beta and plot the results
    for i, beta_value in enumerate(beta):
        D = getD(beta_value, h, k, theta, I0, eta)
        D_dBi = 10 * np.log10(D)  # Convert D to dBi
        maxD = np.max(D_dBi)
        print(f'Maximum Directivity for beta={beta_value} is {maxD:.1f} dBi')
        if beta_value != 0:
            plt.plot(theta * 180 / np.pi, D_dBi, color=colors[i], label=f'$\\Phi_0$={beta_value*h**2/np.pi:.2f}$\pi$, $D_{{max}}$ = {maxD:.1f} dBi', linewidth=2)
    
    # Labels, title, and legend
    plt.xlabel(f'$\\theta')
    plt.xticks(np.arange(0, 180.1, 22.5))
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f$\degree$'))
    
    plt.ylabel('Directivity (dBi)')
    plt.title('Directivity While Adding Quadratic Phase Error (2D For Derek)')
    plt.legend()
    
    # Grid
    plt.grid(True)
    
    # Set x-axis ticks and labels
    plt.xticks(np.arange(0, 180.1, 22.5))
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f°'))
    plt.yticks(np.arange(-80, 20, 10))
    
    plt.show()

# Plot the Maximum Directivity vs. Phi0 and h
def plot_max_directivity_vs_phi0(beta, h, k, theta, I0=1, eta=120*np.pi):
    # Initialize array to store the maximum directivity for each beta
    
    wavelength = 2*np.pi/k

    plt.figure(figsize=(10, 6))
    colours = cm.rainbow(np.linspace(0, 1, len(h)))
    for h0, colour in zip(h, colours):
        max_dBi = []
        # Loop over each beta (related to Phi0) value
        for b in beta:
            # Calculate directivity for all theta at once
            D = getD(b, h0, k, theta, I0, eta)
            D_dBi = 10 * np.log10(D)  # Convert to dBi

            # Find the maximum directivity for this beta
            max_dBi.append(np.max(D_dBi))
        plt.plot(beta * h0**2/np.pi, max_dBi, label=f'h={h0/wavelength:.1f}$\lambda$', color=colour, linewidth=2)
        # Plotting
    
    # plt.plot(beta * h**2/np.pi, max_dBi)
    plt.xlabel('$\Phi_0$')
    plt.ylabel('Maximum Directivity (dBi)')
    plt.title('Maximum Directivity vs. $\Phi_0$')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f$\pi$'))
    plt.show()

# Plot the Maximum Directivity vs. Phi0 and h/lambda
def plot_max_directivity_vs_phi0_3D(beta, h, k, theta, I0=1, eta=120*np.pi):
    # Initialize array to store the maximum directivity for each beta
    wavelength = 2*np.pi/k

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid of beta and h values
    beta_grid, h_grid = np.meshgrid(beta, h)

    # Initialize a 2D array to store the maximum directivity for each pair of beta and h values
    max_dBi = np.zeros_like(beta_grid)
    loopCounter = 0
    for i, h0 in enumerate(h):
        # Loop over each beta (related to Phi0) value
        for j, b in enumerate(beta):
            print(f'Progress: {loopCounter/(len(h)*len(beta))*100:.2f}%')
            # Calculate directivity for all theta at once
            D = getD(b, h0, k, theta, I0, eta)
            D_dBi = 10 * np.log10(D)  # Convert to dBi

            # Find the maximum directivity for this beta
            max_dBi[i, j] = np.nanmax(D_dBi)
            loopCounter += 1
            

    # Create a surface plot
    x = beta_grid * h_grid**2/np.pi
    y = h_grid/wavelength
    surf = ax.plot_surface(x, y, max_dBi, cmap='jet')

    ax.set_xlabel('$\Phi_0$')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.1f}$\pi$'.format(val)))
    
    ax.set_ylabel('h/$\lambda$')
    ax.set_zlabel('Maximum Directivity (dBi)')
    ax.set_title('Maximum Directivity vs. $\Phi_0$ and h/$\lambda$')
    pointSkip = 1
    ax.plot_wireframe(x[::pointSkip, ::pointSkip], y[::pointSkip, ::pointSkip], max_dBi[::pointSkip, ::pointSkip], color='k', linewidth=0.6)
    fig.colorbar(surf, shrink=0.5, aspect=5, label='$D_0$ (dBi)')
    plt.show()

# Antenna Dimensions to sweep over
h = 6 * wavelength
Phi0MaxScaler = 50
Phi0 = np.linspace(0, Phi0MaxScaler / h**2, 17)
beta = Phi0 / h**2

# Plotting Stuff
theta = np.linspace(0.01, np.pi - 0.01, 10000)


#plot_3D_surface_U(k, h, theta, Phi0)
plot_2D_curves_U(k, h, theta, Phi0)

plot_3D_directivity(beta, h, k, theta)
plot_2D_curves_D(k, h, theta, beta)


h = 6 * wavelength
Phi0MaxScaler = 20
Phi0 = np.linspace(0, Phi0MaxScaler / h**2, 100)
beta = Phi0 / h**2

plot_max_directivity_vs_phi0(beta=beta, h=np.linspace(6*wavelength, 20*wavelength,10), k=k, theta=theta)    

h = np.linspace(6*wavelength, 20*wavelength, 50)

# Example call to the plotting function
plot_max_directivity_vs_phi0_3D(beta=beta, h=h, k=k, theta=theta)