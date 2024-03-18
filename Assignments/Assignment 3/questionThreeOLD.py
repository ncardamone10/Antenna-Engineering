'''
This file solves the third question of antennas assignment 3

'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad


# Correctly vectorized version
def getE(theta, N, S, C, wavelength):
    L0 = np.sqrt(S**2 + C**2)
    AR = (2*N + 1) / (2 * N)
    p = (L0/wavelength) / (S/wavelength + AR)
    
    psi = (2 * np.pi / wavelength) * (S * np.cos(theta) - L0 / p)

    # Ensure all operations are compatible with vectorized input
    term1 = np.sin(np.pi / (2 * N))
    term2 = np.cos(theta)
    term3 = np.sin(N * psi / 2) / np.sin(psi / 2)
    
    # Using np.nan_to_num to safely handle divisions that could result in NaN or inf
    return term1 * term2 * np.nan_to_num(term3, nan=0.0)

def getU(theta, N, S, C, wavelength, eta=120*np.pi):
    E = getE(theta, N, S, C, wavelength)
    U = np.abs(E)**2 / (2 * eta)
    return U

# Wrapper function for getU that integrates U * sin(theta)
def integrand(theta, N, S, C, wavelength):
    return getU(theta, N, S, C, wavelength) * np.sin(theta) * 0.5

# Function to perform the integration
def integrate_U(N, S, C, wavelength):
    result, error = quad(integrand, 0, np.pi, args=(N, S, C, wavelength))
    #print(f'The error is {error}')
    return result

def getD(theta, N, S, C, wavelength):
    U = getU(theta, N, S, C, wavelength)

    Uavg = integrate_U(N, S, C, wavelength)
    # theta0 = np.linspace(0, np.pi, 1000)
    # U1 = getU(theta0, N, S, C, wavelength)
    # Uavg = np.mean(U1)
    # Prevent division by zero or very small numbers
    return np.nan_to_num(U / np.where(Uavg == 0, np.inf, Uavg))

def getDmax(N, S, C, wavelength):
    theta = np.linspace(0, np.pi, 1000)
    D = getD(theta, N, S, C, wavelength)
    return np.max(D)

c = 3e8
f = 1e9
wavelength = c / f


C = wavelength
N = 4
alpha = 13*np.pi/180
S = C * np.tan(alpha)
Dmax = getDmax(N, S, C, wavelength)
DmaxTheo = 15 * N * S * C**2 / wavelength**3


# Generate theta values for plotting
theta_plot = np.linspace(0, 2 * np.pi, 1000)

# Calculate U and D for plotting
U_plot = getU(theta_plot, N, S, C, wavelength)
D_plot = getD(theta_plot, N, S, C, wavelength)

# Convert U and D to dB
U_dB = 10 * np.log10(U_plot)
D_dB = 10 * np.log10(D_plot)

# Create polar plots for U and D
fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

# Plot Radiation Intensity in dB
axs[0].plot(theta_plot, U_dB)
axs[0].set_title('Radiation Intensity (dB)')
axs[0].set_theta_zero_location('N')
axs[0].set_theta_direction(-1)
axs[0].set_rlim(-80, 0)


# Plot Directivity in dB
axs[1].plot(theta_plot, D_dB)
axs[1].set_title('Directivity (dB)')
axs[1].set_theta_zero_location('N')
axs[1].set_theta_direction(-1)

plt.show()





'''
Goal: find Dmax(N, S, C, wavelength)
Assume Dmax proportional to N^a, S^b, C^c, wavelength^d, then multiply by some constant
a should be 1, b should be 1, c should be 2, d should be -3, the constant should be 15


'''


# Generate synthetic data within specified constraints
np.random.seed(0)  # For reproducibility
sample_size = 100  # Number of data points

# Generate parameters within specified ranges
N = np.random.randint(4, 6, sample_size)  # N > 3
S_to_C_ratio = np.tan(np.radians(np.random.uniform(12, 14, sample_size)))
C = np.random.uniform(3/4, 4/3, sample_size) * wavelength  # C/wavelength between 3/4 and 4/3
S = S_to_C_ratio * C

# Compute Dmax for each set of parameters
Dmax = np.array([getDmax(n, s, c, wavelength) for n, s, c in zip(N, S, C)])

# Prepare data for regression
X = np.log(np.vstack([N, S, C, wavelength*np.ones(sample_size)]).T)  # Using log to linearize the relationship
y = np.log(Dmax)  # Log of Dmax

# Linear regression
reg = LinearRegression().fit(X, y)
coefficients = reg.coef_
intercept = np.exp(reg.intercept_)  # Convert back from log-space

# Output the results
print("Coefficients (a, b, c, d):", coefficients)
print("Constant:", intercept)