import numpy as np
from scipy import special
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import cm

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

# Calculate Prad using the integral method
def getPradIntegral(k, h, beta, I0=1, eta=120*np.pi): 
    result, _ = quad(integrand, 0, np.pi, args=(k, h, beta, I0, eta), epsabs=1e-22, epsrel=1e-22)
    return result*2*np.pi

# Integrand for the Prad integral (U*sin(theta)
def integrand(theta, k, h, beta, I0=1, eta=120*np.pi):
    return getU(k, h, beta, theta, I0, eta) * np.sin(theta)

# Compare the Prad calculated using the summation method and the integral method
def compareSumAndIntegralAndPlot():
    f = 1e9
    c = 3e8
    wavelength = c/f
    k = 2*np.pi/wavelength
    h = np.linspace(0,1/k,100)
    beta = np.linspace(0,1*k,9)
    loopCounter = 0
    I0 = 1
    eta = 120*np.pi

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Analytical Solution for beta/k=1
    y2 = 1.415 + np.log(2 * k*h / np.pi) - special.sici(4 * k*h)[1] + np.sin(4 * k*h) / (4 * k*h + 1e-20)
    y2 = y2 * eta * I0**2 / (4 * np.pi)
    axs[0].plot(h*k, y2, label='Analytical Solution for $\\beta/k$=1', color='black', linewidth=6)

    colors = cm.rainbow(np.linspace(0, 1, len(beta)))

    for beta0, color in zip(beta, colors):
        PradSum = []
        PradIntegral = []
        for h0 in h:
            print(f'Calculating Prad for h0={h0} and beta0={beta0}, Progress: {loopCounter/(len(h)*len(beta))*100:.2f}%')
            PradSum.append(getPradSum(k, h0, beta0, I0, eta))
            PradIntegral.append(getPradIntegral(k, h0, beta0, I0, eta))
            loopCounter += 1
        axs[0].plot(h*k, PradSum, color=color, label=f'$\\beta/k$={beta0/k} (Sum)', linewidth=2)
        axs[0].plot(h*k, PradIntegral, '--', color=color, label=f'$\\beta/k$={beta0/k} (Integral)', linewidth=2)
        axs[1].plot(h*k, 10*np.log10(np.array(PradSum) / np.array(PradIntegral)), color=color, label=f'$\\beta/k$={beta0/k}', linewidth=2)

    axs[0].set_xlabel(r'$kh$')
    axs[0].set_ylabel(r'$P_{rad}$')
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title('Comparison of $P_{rad}$ calculated using Summation (M.Dich) and Integration of Usin($\\theta$) methods')

    axs[1].set_xlabel(r'$kh$')
    axs[1].set_ylabel(r'$\frac{P_{rad,Sum}}{P_{rad,Integral}}$ (dB)')
    axs[1].legend()
    axs[1].grid()
    axs[1].set_ylim(-0.005, 0.005)
    axs[1].set_title('Comparison of $P_{rad}$ Error (dB)')

    plt.tight_layout()
    plt.show()
            
compareSumAndIntegralAndPlot()          






