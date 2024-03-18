'''
This file solves the second question of antennas assignment 3
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.integrate import simpson
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


f = 1e9
c = 3e8
k = 2*np.pi*f/c
wavelength = c/f



# Directivity for the antenna (using U and weighted average)
def getD(k, h, beta, theta):
    eta = 120 * np.pi
    I0 = 1
    #E = getE(k, h, beta, theta, I0, eta)

    # Check that theta is between 0 and pi
    assert(min(theta) == 0 and max(theta) == np.pi)

    # Calculate radiation intensity U
    U = getU2(k, h, beta, theta, I0, eta)

    # Generate theta values for integration
    sinTheta = np.sin(theta)  # Compute sin(theta) for weighting
   
    # Compute average U using weights
    #Uavg = np.average(U, weights=sinTheta)
    Uavg = 0.5 * simpson(U*sinTheta, theta)
    # Calculate directivity D
    D = U / Uavg

    return D

# Directivity for the antenna (analytical solution)
def getD2(k, h, beta, theta, eta=120*np.pi):
    # Check that theta is between 0 and pi
    assert(min(theta) == 0 and max(theta) == np.pi)    
    # Denominator terms
    term1_denom = 1.415
    term2_denom = np.log(2 * k * h / np.pi)
    term3_denom = -special.sici(4 * k * h)[1]
    term4_denom = np.sin(4 * k * h) / (4 * k * h + 1e-22)
    
    # Combine the denominator terms
    denom = term1_denom + term2_denom + term3_denom + term4_denom
    #denom *= (2 * h)**2
    
    # Alpha calculation
    alpha = np.cos(theta) - beta / k
    
    # Numerator terms
    term1_num = 2  # Constant term in the numerator
    term2_num = np.sin(theta)**2  # Depends on theta
    term3_num = (np.sin(k * h * alpha)**2) / ((alpha + 1e-20)**2)  # Depends on alpha
    
    # Combine the numerator terms
    num = term1_num * term2_num * term3_num
    
    # Directivity calculation
    D = num / denom
    
    return D

# Max Directivity of the helical antenna (returns [angle (degrees), Dmax])
def getDmax(k, h, beta):
    # Generate theta values in radians
    theta = np.linspace(0, np.pi, 500)
    
    # Compute U for all theta values at once
    U = getU2(k=k, h=h, beta=beta, theta=theta)
    
    # Compute weights for averaging U
    sinTheta = np.sin(theta)
    
    # Calculate average U using weights
    Uavg = np.average(U, weights=sinTheta)
    
    # Calculate directivity D
    D = U / Uavg
    
    # Find the index of maximum directivity
    maxIndex = np.argmax(D)
    
    # Convert the angle of maximum directivity to degrees and return it with the max D value
    return theta[maxIndex] * 180 / np.pi, np.max(D)

# Radiation Intensity for the antenna (analytical solution)
def getU(k, h, beta, theta, I0=1, eta=120*np.pi):
    # Calculate alpha
    alpha = np.cos(theta) - beta / k

    # Calculate the radiation intensity (U) using the given terms
    term1 = I0**2 * eta / (8 * np.pi**2)
    term2 = np.sin(theta)**2
    term3 = np.sin(k * h * alpha)**2 / (alpha + 1e-12)**2

    # Compute and return the product of the terms for U
    U = term1 * term2 * term3 #/ (2 * h)**2
    return U

# Radiation Intensity for the antenna (from E field)
def getU2(k, h, beta, theta, I0=1, eta=120*np.pi):
    return np.abs(getE(k, h, beta, theta, I0, eta))**2 / (2*eta)

# Electric field for the antenna
def getE(k, h, beta, theta, I0=1, eta=120*np.pi):
    # Calculate alpha
    alpha = np.cos(theta) - beta / k

    # Calculate the electric field (E) using the given terms
    # The operations are vectorized, allowing 'theta' to be an array or a scalar
    E = 1j * eta * k * I0 / (4 * np.pi) * 2 * h
    E *= np.sin(theta)
    E *= np.sin(k * h * alpha)
    E /= (k * h * alpha + 1e-10)  # Adding a small value to avoid division by zero

    return E

# Check to see if Prad and 4piUavg are the same
def comparePradWithUavg(k, beta):
    I0 = 1
    eta = 120 * np.pi
    numberOfPoints = 741
    # Correction: Ensure L calculation is consistent and doesn't result in a sequence error
    L = np.linspace(0, 20.05*2*np.pi/k, numberOfPoints)
    
    # Generate theta values in radians
    theta = np.linspace(0, np.pi, 500)
    # Compute weights for averaging U
    sinTheta = np.sin(theta)

    Prad = []
    Uavg = []
    for L0 in L:
        U = getU(k=k, h=L0/2, beta=beta, theta=theta, I0=I0, eta=eta)
        UavgTemp = np.average(U, weights=sinTheta)
        #UavgTemp = 0.5 * simpson(U*sinTheta, theta)
        Uavg.append(UavgTemp)

        PradTemp1 = eta * I0**2 / (4 * np.pi)
        PradTemp2 = 1.415 + np.log(2 * k * (L0/2) / np.pi) - special.sici(4 * k * (L0/2))[1] + np.sin(4 * k * (L0/2)) / (4 * k * (L0/2) + 1e-12)
        Prad.append(PradTemp1 * PradTemp2)

    # Plot Prad and Uavg as a function of L/lambda
    plt.figure()
    plt.plot(L/(2*np.pi/k), Prad, label='Prad')
    plt.plot(L/(2*np.pi/k), 4*np.pi*np.array(Uavg), label='$4\pi$Uavg')  # Ensure Uavg is a NumPy array for multiplication
    plt.xlabel('$L/\lambda$')
    plt.ylabel('Prad and $4\pi$Uavg')
    plt.title('Prad and $4\pi$Uavg as a function of $L/\lambda$')
    plt.grid(True)
    plt.legend()
    plt.show()

# Compare the directivity, radiation intensity, and magnitude of E for different values of k, h, and beta
def plot_antenna_characteristics(k, h, beta):
    # Original theta values in radians for 0 to 180 degrees
    theta_positive = np.linspace(0, np.pi, 500)
    # Mirror theta for -180 to 0 degrees
    theta_negative = np.linspace(-np.pi, 0, 500)
    # Combine for full range of -180 to 180 degrees
    theta = np.concatenate((theta_negative, theta_positive))
    
    # Calculate characteristics for positive theta values
    D_positive = getD(k, h, beta, theta_positive)
    U_positive = getU(k, h, beta, theta_positive)
    U2_positive = getU2(k, h, beta, theta_positive)
    E_positive = getE(k, h, beta, theta_positive)
    D2_positive = getD2(k, h, beta, theta_positive)  # Calculate directivity using getD2

    # Mirror the characteristics for negative theta by simply duplicating the positive values
    D = np.concatenate((D_positive, D_positive))
    U = np.concatenate((U_positive, U_positive))
    U2 = np.concatenate((U2_positive, U2_positive))
    E = np.concatenate((E_positive, E_positive))
    D2 = np.concatenate((D2_positive, D2_positive))  # Directivity from getD2

    # Convert to dB
    D_dB = 10 * np.log10(D)
    U_dB = 10 * np.log10(U)
    U2_dB = 10 * np.log10(U2)
    E_dB = 20 * np.log10(np.abs(E))  # Magnitude of E in dB
    D2_dB = 10 * np.log10(D2)  # Directivity from getD2 in dB

    # Plot on polar subplots
    fig, axs = plt.subplots(1, 5, subplot_kw={'projection': 'polar'}, figsize=(25, 5))
    
    plot_characteristics(axs[0], theta, D_dB, 'Directivity in dB (Using Weighted Avg on U)')
    plot_characteristics(axs[1], theta, D2_dB, 'Directivity in dB (Analytical Solution)')
    plot_characteristics(axs[2], theta, U_dB, 'Radiation Intensity (Analytical) in dB')
    plot_characteristics(axs[3], theta, U2_dB, 'Radiation Intensity (From E field) in dB')
    plot_characteristics(axs[4], theta, E_dB, 'Magnitude of E in dB')
      

    plt.tight_layout()
    plt.show()

def plot_characteristics(ax, theta, values, title):
    ax.plot(theta, values, label=title.split(' ')[0])
    ax.set_title(title)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(-40, 1.1*max(values))  # Set dB scale from -40 to 0 dB

# Compare Given Prad with (equivalent?) numerical integral
def plotIntegral():
    # Parameters for plotting
    b_values = [0, 0.5, 1]  # Example b values
    kh_range = np.linspace(0, 20, 20)  # kh range

    plt.figure(figsize=(10, 7))
    count = 0
    maxCount = len(b_values) * len(kh_range)

    for b in b_values:
        y_values = []
        for kh in kh_range:
            result = integral_y(kh, b)
            y_values.append(result)
            percentDone = count / maxCount * 100
            count += 1
            print(f'kh: {kh}, b: {b}, result: {result}, Progress: {percentDone:.2f}%')
        #y_values = [integral_y(kh, b) for kh in kh_range]
        plt.plot(kh_range, y_values, label=f'Numerical Solution, b/k={b}')

    y2 = 1.415 + np.log(2 * kh_range / np.pi) - special.sici(4 * kh_range)[1] + np.sin(4 * kh_range) / (4 * kh_range + 1e-20)
    plt.plot(kh_range, y2, label='Analytical Solution')
    plt.xlabel('kh')
    plt.ylabel('Integral Value')
    plt.title('Plot of the Integral vs. kh')
    plt.legend()
    plt.grid(True)
    plt.show()

# (Equivalent?) Numerical Integral (to compare with Prad)
def integrand(theta, kh, bk):
    return (np.sin(theta)**2) * (np.sin(kh * (np.cos(theta) - bk))**2) / ((np.cos(theta) - bk)**2 + 1e-24)

# Function with gaussian quadrature to integrate the integrand
def integral_y(kh, bk):
    # Using Gaussian quadrature to integrate the integrand
    # result, _ = quad(integrand, 0, np.pi, args=(kh, bk), epsabs=1e-20, epsrel=1e-20)
    # return result

    # # Using Simpson's rule to integrate the integrand
    # x = np.linspace(0, np.pi, 100000)  # define x-values
    # y = integrand(x, kh, bk)  # compute y-values
    # result = simpson(y, x)  # apply Simpson's rule
    # return result

    x = np.linspace(0, np.pi, 100000001)  # define x-values
    y = integrand(x, kh, bk)  # compute y-values

    # Break up x and y into 1000 partitions
    x_partitions = np.array_split(x, 10000)
    y_partitions = np.array_split(y, 10000)

    partialSum = 0
    loopCount = 0

    # Integrate each partition and sum the results
    for y_part, x_part in zip(y_partitions, x_partitions):
        result = simpson(y_part, x_part)
        partialSum += result
        #print(f'Loop: {loopCount}, result: {result}')
        loopCount += 1
    #result = (simpson(y_part, x_part) for y_part, x_part in zip(y_partitions, x_partitions))
    return partialSum



plotIntegral()

comparePradWithUavg(k, 0)

plot_antenna_characteristics(k, 1*wavelength, 0)





# Plot max directivity as a function of L (Method A)
numberOfPoints = 741
L = np.linspace(0, 20.05*wavelength, numberOfPoints)
theta = np.linspace(0, 180, 360)*np.pi/180
sinTheta = np.sin(theta)


# Plot max directivity as a function of L (Method A: using the getDmax function)
DmaxA = []
thetaMax = []
for L0 in L:
    # U = getU(k=k, h=0.5*L0, beta=0, theta=theta)
    # Uavg = np.average(U, weights=sinTheta)
    # D = U/Uavg
    maxTheta, D = getDmax(k=k, h=L0/2, beta=0)
    DmaxA.append(D)
    thetaMax.append(maxTheta*np.pi/180)
    #print(f'Dmax for L0={L0/wavelength} lambda is {np.max(D)}')
DmaxA = np.array(DmaxA)
thetaMaxA = np.array(thetaMax)

# Plot max directivity as a function of L (Method B: using the getD2 function which is the analytical solution)
D = []
thetaMax = []
for L0 in L:
    Dtemp = getD2(k=k, h=L0/2, beta=0, theta=theta)
    D.append(np.max(Dtemp))
    maxIndex = np.argmax(Dtemp)
    thetaMax.append(theta[maxIndex])
DmaxB = np.array(D)
thetaMaxB = np.array(thetaMax)

# Plot max directivity as a function of L (Method C: Integration of U*sin(theta) using simpson for Uavg)
DmaxC = []
thetaMax = []
for L0 in L:
    U = getU(k=k, h=0.5*L0, beta=0, theta=theta) 
    Uavg = 0.5 * simpson(U*np.sin(theta), theta)
    D = U/Uavg
    DmaxC.append(np.max(D))
    maxIndex = np.argmax(D)
    thetaMax.append(theta[maxIndex])
    #print(f'Dmax for L0={L0/wavelength} lambda is {np.max(D)}')
DmaxC = np.array(DmaxC)
thetaMaxC = np.array(thetaMax)


# Start plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Creates a figure and 1x2 grid of subplots

# Plot 1: Max Directivity as a function of L
axs[0].plot(L/wavelength, 10 * np.log10(DmaxA), label='Method A')
axs[0].plot(L/wavelength, 10 * np.log10(DmaxB), label='Method B')
axs[0].plot(L/wavelength, 10 * np.log10(DmaxC), label='Method C')
axs[0].set_xlabel('$L/\\lambda$')
axs[0].set_ylabel('Max Directivity (dBi)')
axs[0].set_title('Max Directivity vs. $L/\\lambda$')
axs[0].grid(True)
axs[0].legend()

# Convert thetaMax from radians to degrees for plotting
thetaMaxA_deg = np.rad2deg(thetaMaxA)
thetaMaxB_deg = np.rad2deg(thetaMaxB)
thetaMaxC_deg = np.rad2deg(thetaMaxC)

# Plot 2: Theta at Max Directivity as a function of L
axs[1].plot(L/wavelength, thetaMaxA_deg, label='Method A')
axs[1].plot(L/wavelength, thetaMaxB_deg, label='Method B')
axs[1].plot(L/wavelength, thetaMaxC_deg, label='Method C')
axs[1].set_xlabel('$L/\\lambda$')
axs[1].set_ylabel('Theta at Max Directivity (degrees)')
axs[1].set_title('Theta at Max Directivity vs. $L/\\lambda$')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()  # Adjusts the subplots to fit into the figure area
plt.show()


L0 = [5*wavelength, 10*wavelength, 20*wavelength, 40*wavelength]
plt.figure()

# Loop through L0 and plot U for all the L0 values
for L0 in L0:
    # Radiation Intensity for the antenna
    U = []
    for theta0 in theta:
        U.append(getU2(k=k, h=L0/2, beta=0, theta=theta0))
    U = np.array(U)
    U = U/np.max(U)

    # Plotting the radiation intensity
    plt.plot(theta*180/np.pi, 10*np.log10(U), label=f'$L/\lambda={L0/wavelength} $')


plt.legend()
plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel('Radiation Intensity (dBi)')
plt.title('Radiation Intensity of the antenna')
plt.grid(True)
plt.ylim(-40, 0)
# Set the x-axis ticks to be multiples of 45 degrees
plt.xticks(np.arange(0, 181, 22.5))
plt.show()




# Now make a plot of the beam angle as a function of beta
betaRange = 1
beta = np.linspace(-k*betaRange, k*betaRange, 100)
theta = np.linspace(0, 180, numberOfPoints) * np.pi / 180

# Define the range of L values
L_values = [6*wavelength, 9*wavelength, 12*wavelength, 15*wavelength, 18*wavelength, 21*wavelength]

# Plotting for the current value of L
fig, axs = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
#fig.suptitle(f'Plots for L = {L/wavelength} $\lambda$')

for L in L_values:
    thetaS = []
    Dmax = []

    for beta0 in beta:
        # Get the radiation intensity
        #U = getU(k=k, h=L/2, beta=beta0, theta=theta)
        
        # # Find the max radiation intensity and the angle at which it occurs
        # maxUIndex = np.argmax(U)
        # maxTheta = theta[maxUIndex] * 180 / np.pi  # Convert to degrees

        D = getD2(k=k, h=L/2, beta=beta0, theta=theta)
        D0 = np.max(D)
        maxIndex = np.argmax(D)
        maxTheta = theta[maxIndex] * 180 / np.pi
        # D0 = getDmax(k=k, h=L/2, beta=beta0)
        # maxTheta, D0 = getDmax(k=k, h=L/2, beta=beta0)

        # Save the results
        thetaS.append(maxTheta)
        Dmax.append(D0)


    # Plot thetaS as a function of beta/k
    axs[0].plot(beta/k, thetaS, label=f'L = {L/wavelength} $\lambda$')
    

    # Plot Dmax as a function of thetaS
    axs[1].plot(thetaS, 10*np.log10(Dmax), label=f'L = {L/wavelength} $\lambda$')



axs[0].set_xlabel(r'$\frac{\beta}{k}$')
axs[0].set_ylabel(r'$\theta_s$')
axs[0].set_title('Beam angle as a function of $\\beta$')
axs[0].grid(True)
axs[0].set_yticks(np.arange(0, 181, 22.5))
axs[1].set_xlabel(r'$\theta_s$')
axs[1].set_ylabel(r'$D_{max}$ (dBi)')
axs[1].set_title('Max directivity as a function of $\\theta_s$')
axs[1].grid(True)
axs[1].set_xticks(np.arange(0, 181, 22.5))
axs[0].legend()
axs[1].legend()

plt.show()









