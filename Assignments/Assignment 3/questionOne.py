import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load the data from the spreadsheet
file_path = 'C:\\Users\\ncard\\OneDrive - University of Ottawa\\University Archive\\Masters\\Antenna-Engineering\\Assignments\\Assignment 3\\Field Data for Helix Antenna in Phi = 0 Pattern Cut.xlsx'
data = pd.read_excel(file_path)

# Convert theta from degrees to radians for polar plotting
theta_radians = np.deg2rad(data['Theta (Degrees)'])

# Calculate the total magnitude squared for each theta for the radiation intensity
magnitude_squared = data['E-Phi (Magnitude)']**2 + data['E-Theta (Magnitude)']**2
max_magnitude_squared = np.max(magnitude_squared)
radiation_intensity_dB = 10 * np.log10(magnitude_squared / max_magnitude_squared)

# Calculate the signed AR (not in dB)
#signed_ar = data['E-Theta (Magnitude)'] / data['E-Phi (Magnitude)']
Etheta = data['E-Theta (Magnitude)'] * np.exp(1j * data['E-Theta (Phase in Degrees)'] * np.pi / 180)
Ephi = data['E-Phi (Magnitude)'] * np.exp(1j * data['E-Phi (Phase in Degrees)'] * np.pi / 180)

rhoLinear = Ephi / Etheta
rhoCircular = ( 1 + 1j * rhoLinear ) / ( 1 - 1j * rhoLinear )

signed_ar = ( np.abs(rhoCircular) + 1 ) / ( np.abs(rhoCircular) - 1 )

# Calculate the AR in dB
ar_dB = 20 * np.log10(abs(signed_ar))

# Now go for the directivity
# Calculate U for each theta
U = (data['E-Theta (Magnitude)']**2 + data['E-Phi (Magnitude)']**2)/(2*377)

# Determine the midpoint of the array, assuming an even number of points covering -180 to 180 degrees
midpoint = np.abs(data['Theta (Degrees)'] - 0).argmin()

# Approach 1: Direct using mean of U
average_U1 = np.mean(U[midpoint:])
D1 = U / average_U1
D1_dBi = 10 * np.log10(D1)

print(f'Average U (Method 1): {average_U1}')

# Approach 2: Direct using integral to find average U
# Slice the arrays to take only the upper half
theta_degrees_half = data['Theta (Degrees)'][midpoint:].reset_index(drop=True)
E_theta_half = data['E-Theta (Magnitude)'][midpoint:].reset_index(drop=True)
E_phi_half = data['E-Phi (Magnitude)'][midpoint:].reset_index(drop=True)
U_half = (E_theta_half**2 + E_phi_half**2)/(2*377)

# Recalculate U*sin(theta) using the sliced arrays
theta_radians_half = theta_degrees_half * np.pi / 180
U_sin_theta_half = U_half * np.sin(theta_radians_half)

# Proceed with average U calculation and directivity calculation using the sliced arrays
average_U2_half_upper = 0.25 * simpson(U_sin_theta_half, theta_radians_half)
# D2_half = U_half / average_U2_half
# D2_dBi_Upper = 10 * np.log10(D2_half)
print(f'Average U (Method 2 Upper): {average_U2_half_upper}')

# Slice the arrays to take only the lower half
theta_degrees_half = data['Theta (Degrees)'][0:midpoint].reset_index(drop=True)
E_theta_half = data['E-Theta (Magnitude)'][0:midpoint].reset_index(drop=True)
E_phi_half = data['E-Phi (Magnitude)'][0:midpoint].reset_index(drop=True)
U_half = (E_theta_half**2 + E_phi_half**2)/(2*377)

# Recalculate U*sin(theta) using the sliced arrays
theta_radians_half = theta_degrees_half * np.pi / 180
U_sin_theta_half = U_half * np.sin(theta_radians_half)

# Proceed with average U calculation and directivity calculation using the sliced arrays
average_U2_half_lower = 0.25 * simpson(np.flip(U_sin_theta_half), np.flip(theta_radians_half))
# D2_half = U_half / average_U2_half
# D2_dBi_Lower = 10 * np.log10(D2_half)
print(f'Average U (Method 2 Lower): {average_U2_half_lower}')

#D2_dBi = np.concatenate((D2_dBi_Lower, D2_dBi_Upper))
D2 = U / (average_U2_half_upper + average_U2_half_lower)
D2_dBi = 10 * np.log10(D2)
assert(len(D2_dBi) == len(D1_dBi))

# Approach 3: Using a weighted average of U and sin(theta)
sintheta = np.sin(data['Theta (Degrees)'][midpoint:] * np.pi / 180)
Uwavg = np.average(U[midpoint:], weights=sintheta)
D3 = U / Uwavg
D3_dBi = 10 * np.log10(D3)

# Compare the max directivity for each method
print(f'The max directivity in dBi for Method 1 is {max(D1_dBi)} dBi')
print(f'The max directivity in dBi for Method 2 is {max(D2_dBi)} dBi')
print(f'The max directivity in dBi for Method 2 is {max(D3_dBi)} dBi')
print(f'The 1-2 difference is {max(D1_dBi) - max(D2_dBi)} dBi or {10**(max(D2_dBi)/10) / 10**(max(D1_dBi)/10)} times')
print(f'The 2-3 difference is {max(D2_dBi) - max(D3_dBi)} dBi or {10**(max(D3_dBi)/10) / 10**(max(D2_dBi)/10)} times')

# Plotting all three in subplots with constrained layout
fig, axs = plt.subplots(4, 1, figsize=(10, 18), constrained_layout=True)

# Function to set 10 degree increments for x-ticks and add degree symbols
def set_xticks_22_5(ax):
    ax.set_xticks(np.arange(-180, 181, 10))
    ax.set_xlim(-180, 180)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

# Create a function to format the ticks
def format_func(value, tick_number):
    # convert radians to degrees and add degree symbol
    return f'${value}\degree$'


# Plot 1: Normalized Radiation Intensity (dBi)
axs[0].plot(data['Theta (Degrees)'], radiation_intensity_dB, label='Normalized Radiation Intensity (dBi)')
axs[0].set_xlabel('$\\theta$  ')
axs[0].set_ylabel('Intensity (dBi)')
axs[0].set_title('Normalized Radiation Intensity of the Helical Antenna')
set_xticks_22_5(axs[0])
axs[0].grid(True)
#axs[0].legend()

# Plot 2: Signed AR (not in dB)
axs[1].plot(data['Theta (Degrees)'], signed_ar, label='Signed AR', color='orange')
axs[1].set_xlabel('$\\theta$  ')
axs[1].set_ylabel('Axial Ratio (Linear Units)')
axs[1].set_title('Signed Axial Ratio (Linear Units)')
set_xticks_22_5(axs[1])
axs[1].grid(True)
#axs[1].legend()
axs[1].set_ylim(-15, 15)

# Plot 3: AR in dB
axs[2].plot(data['Theta (Degrees)'], ar_dB, label='AR (dB)', color='green')
axs[2].set_xlabel('$\\theta$  ')
axs[2].set_ylabel('Axial Ratio (dB)')
axs[2].set_title('Axial Ratio (dB)')
set_xticks_22_5(axs[2])
axs[2].grid(True)
#axs[2].legend()


# Plot 4: Directivity (dBi)
axs[3].plot(data['Theta (Degrees)'], D1_dBi, label=f'Method 1 : Discrete Average, $D_{{max}}$ = {max(D1_dBi):.1f} dBi', color='red')
axs[3].plot(data['Theta (Degrees)'], D2_dBi, label=f'Method 2 : Average via Integration, $D_{{max}}$ = {max(D2_dBi):.1f} dBi', color='magenta')
axs[3].scatter(data['Theta (Degrees)'], D3_dBi, label=f'Method 3 : Weighted Discrete Average, $D_{{max}}$ = {max(D3_dBi):.1f} dBi', color='black', s=10)
axs[3].set_xlabel('$\\theta$  ')
axs[3].set_ylabel('Directivity (dBi)')
axs[3].set_title('Directivity of the Helical Antenna (dBi)')
axs[3].grid(True)
set_xticks_22_5(axs[3])
axs[3].legend()


# Show plot
plt.show()

# Start a new figure for polar plots
fig, axs = plt.subplots(1, 3, figsize=(14, 7), subplot_kw={'projection': 'polar'}, constrained_layout=True)

# Plot 1: Normalized Radiation Intensity (dBi) on a polar plot
axs[0].plot(theta_radians, radiation_intensity_dB, label='Normalized Radiation Intensity (dBi)')
axs[0].set_title('Radiation Intensity (dBi)')
axs[0].set_theta_zero_location('N')  # Set the direction of 0 degrees to the top
axs[0].set_theta_direction(-1)  # Set the theta direction to clockwise
axs[0].set_ylim(min(radiation_intensity_dB), max(radiation_intensity_dB))  # Adjust the radial limits to fit the data
axs[0].grid(True)
#axs[0].legend(loc='upper right')

# Plot 2: Axial Ratio (dB) on a polar plot
axs[1].plot(theta_radians, ar_dB, label='Axial Ratio (dB)', color='green')
axs[1].set_title('Axial Ratio (dB)')
axs[1].set_theta_zero_location('N')  # Set the direction of 0 degrees to the top
axs[1].set_theta_direction(-1)  # Set the theta direction to clockwise
axs[1].set_ylim(min(ar_dB), max(ar_dB))  # Adjust the radial limits to fit the data
axs[1].grid(True)
#axs[1].legend(loc='upper right')

# Plot 3: Directivity (dBi) on a polar plot
axs[2].plot(theta_radians, D1_dBi, label=f'Method 1 : Discrete Average, $D_{{max}}$ = {max(D1_dBi):.1f} dBi', color='red')
axs[2].plot(theta_radians, D2_dBi, label=f'Method 2 : Average via Integration, $D_{{max}}$ = {max(D2_dBi):.1f} dBi', color='magenta')
axs[2].scatter(theta_radians, D3_dBi, label=f'Method 3 : Weighted Discrete Average, $D_{{max}}$ = {max(D3_dBi):.1f} dBi', color='black', s=10)
axs[2].set_title('Directivity (dBi)')
axs[2].grid(True)
axs[2].set_theta_zero_location('N')  # Set the direction of 0 degrees to the top
axs[2].set_theta_direction(-1)  # Set the theta direction to clockwise
#axs[1].set_ylim(min(ar_dB), max(ar_dB))  # Adjust the radial limits to fit the data
axs[2].legend()

# Show the polar plots
plt.show()


