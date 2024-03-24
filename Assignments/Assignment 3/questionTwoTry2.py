''' 
This solves question 2 of the assignment.
'''

import numpy as np
from scipy import special
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mticker

f = 1e9
c = 3e8
k = 2*np.pi*f/c
wavelength = c/f

# Electric field for a traveling wave antenna
def getE(k, h, beta, theta, I0=1, eta=120*np.pi):
    # Calculate alpha
    alpha = np.cos(theta) - beta / k

    # Calculate the electric field (E) using the given terms
    # The operations are vectorized, allowing 'theta' to be an array or a scalar
    E = 1j * eta * k * I0 / (4 * np.pi) * 2 * h
    E *= np.sin(theta)
    E *= np.sin(k * h * alpha)
    E /= (k * h * alpha + 1e-22)  # Adding a small value to avoid division by zero

    return E
 
# Radiation Intensity for the antenna (from E field)
def getU(k, h, beta, theta, I0=1, eta=120*np.pi):
    return np.abs(getE(k, h, beta, theta, I0, eta))**2 / (2*eta)
      
# Calculate Prad using the summation method given my Mikael Dich (Oct 1997)    
def getPradSum(wavenumber, h, beta, I0=1, eta=120*np.pi):
    wavelength = 2*np.pi/wavenumber
    N = int(2*np.pi*h/wavelength) + 10
    K = (2*N + 1)
    L = (4*N + 1)
    #print(f'K={K}, L={L}, N={N}')
    
    eps = lambda i: 1 if i == 0 else 2 
    
    kPartialSum = 0
    for k in range(0, K+1): # Starts at 0, ends at K
        lPartialSum = 0
        for l in range(0,L): # Starts at 0, ends at L-1
            lPartialSum += getU(wavenumber, h, beta, np.pi*k/K, I0, eta)
        iPartialSum = 0
        for i in range(0, int(K/2)+1): # Starts at 0, ends at K/2
            iPartialSum += eps(i) * (2 * np.cos(2*np.pi*i*k/K))/(1 - 4*i**2)
        kPartialSum += lPartialSum * iPartialSum * eps(k) * eps(K - k)
            
    return kPartialSum * np.pi / ( 2 * K * L)

# Directivity using Prad from the summation method (any beta/k will work) (and numerical calculation of U from E)
def getD(k, h, beta, theta, I0=1, eta=120*np.pi):
    #theta = np.linspace(0, np.pi, 1000)
    U = getU(k, h, beta, theta, I0, eta)
    Prad = getPradSum(k, h, beta, I0, eta)
    return 4*np.pi*U/Prad

# Plot Dmax vs antenna electrical size (answers part c)
def plotDmaxOverAntennaSize(Lmin=0, Lmax=20, wavelength=wavelength, betaNorm=[0], I0=1, eta=120*np.pi):
    L = np.linspace(Lmin*wavelength, Lmax*wavelength, 100)
    k = 2*np.pi/wavelength
    beta = [beta0 * k for beta0 in betaNorm]
    theta = np.linspace(0, np.pi, 1000)
    loopCounter = 0
    colors = plt.cm.rainbow(np.linspace(0, 1, len(beta)))
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    for beta0, color in zip(beta, colors):
        Dmax = np.zeros(L.size)
        thetaMax = np.zeros(L.size)
        for i in range(L.size):
            print(f'Calculating Dmax for beta0={beta0}, Progress: {loopCounter/(L.size*len(beta))*100:.2f}%')
            D = getD(k, L[i]/2, beta0, theta, I0, eta)
            Dmax[i] = np.max(D)
            thetaMaxIdx = np.argmax(D)
            thetaMax[i] = theta[thetaMaxIdx]*180/np.pi
            loopCounter += 1
        axs[0].plot(L/wavelength, 10*np.log10(Dmax), color=color, label=f'$\\beta/k$={beta0/k}')
        axs[1].plot(L/wavelength, thetaMax, color=color, label=f'$\\beta/k$={beta0/k}')
    
    axs[0].set_xlabel('$\\frac{L}{\lambda}$')
    axs[0].set_ylabel('Max Directivity (dBi)')
    axs[0].set_title('Max Directivity vs Electrical Length of a Traveling Wave Antenna')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_xlabel('$\\frac{L}{\lambda}$')
    axs[1].set_ylabel('$\\theta_{max}$ $(\degree)$')
    axs[1].set_title('$\\theta_{max}$ vs Electrical Length of a Traveling Wave Antenna')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()

# Plot Normalized Radiation Intensity over theta for different antenna sizes (answers part d)
def plotUPatternOverAntennaSize(theta, k=k, h=[1*wavelength], beta=0, I0=1, eta=120*np.pi):
    wavelength = 2*np.pi/k
    plt.figure()
    # Loop through L0 and plot U for all the L0 values
    for h0 in h:
        # Radiation Intensity for the antenna
        U = getU(k, h0, beta, theta, I0, eta)
        U = U/np.max(U)
        # Plotting the radiation intensity
        plt.plot(theta*180/np.pi, 10*np.log10(U), label=f'$L/\lambda={2*h0/wavelength} $')
    plt.legend()
    plt.xlabel(f'$\\theta$')
    plt.ylabel('Normalized Radiation Intensity (dBi)')
    plt.title(f'Normalized Radiation Intensity of the Traveling Wave Antenna for $\\beta/k={beta/k}$')
    plt.grid(True)
    plt.ylim(-40, 0)
    # Set the x-axis ticks to be multiples of 45 degrees
    plt.xticks(np.arange(0, 181, 22.5))
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x}°'))
    plt.show()

# Scan the beam and plot Dmax and thetaMax for different antenna sizes (answers part f)
def scanTheBeamAndPlotDmax(k=k, h=[1*wavelength], beta=[0*k], I0=1, eta=120*np.pi):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    loopCounter = 0
    # Create a colormap
    colors = cm.rainbow(np.linspace(0, 1, len(h)))
    # First, we need to loop through the different antenna sizes
    for h0, color in zip(h, colors):
        # Ok so now lets vary beta and find the Dmax and thetaMax for each beta
        Dmax = []
        thetaMax = []
        for beta0 in beta:
            theta = np.linspace(0, np.pi, 1000)
            D = getD(k, h0, beta0, theta, I0, eta)
            Dmax.append(np.max(D))
            thetaMaxIdx = np.argmax(D)
            thetaMax.append(theta[thetaMaxIdx]*180/np.pi)
            print(f'Progress: {loopCounter/(len(beta)*len(h))*100:.2f}%')
            loopCounter += 1
        axs[0].plot(beta/k, thetaMax, label=f'$L/\lambda={2*h0/wavelength:.2f}$', color=color, linewidth=2)
        axs[1].plot(thetaMax, 10*np.log10(Dmax), label=f'$L/\lambda={2*h0/wavelength:.2f}$', color=color, linewidth=2)
    axs[0].set_xlabel('$\\beta/k$')
    axs[0].set_ylabel('$\\theta_s$')
    axs[0].set_title('Beam Angle vs $\\beta/k$')
    axs[0].legend()
    axs[0].grid()
    axs[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x}°'))

    axs[1].set_xlabel('$\\theta_s$')
    axs[1].set_ylabel('Max Directivity (dBi)')
    axs[1].set_title('Max Directivity vs Beam Angle')
    axs[1].legend(loc='lower right')
    axs[1].grid()
    axs[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x}°'))

    plt.tight_layout()
    plt.show()

# Part C
plotDmaxOverAntennaSize(Lmin=0, Lmax=20, wavelength=wavelength, betaNorm=np.linspace(0,1,9), I0=1, eta=120*np.pi)

# Part D
# plotUPatternOverAntennaSize(theta=np.linspace(0, np.pi, 1000), k=k, h=[2.5*wavelength, 5*wavelength, 10*wavelength, 20*wavelength], beta=0.0*k, I0=1, eta=120*np.pi)

# Part F
#scanTheBeamAndPlotDmax(k=k, h=np.linspace(3*wavelength, 10*wavelength, 11), beta=np.linspace(0, 1*k, 100), I0=1, eta=120*np.pi)


