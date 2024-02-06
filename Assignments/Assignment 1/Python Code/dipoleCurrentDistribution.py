import numpy as np
import matplotlib.pyplot as plt

# Constants
I_0 = 1
f = 2e9 # 2 GHz Dipole
c = 3e8 # speed of light
wavelength = c / f
k = 2 * np.pi / wavelength
eta = 120 * np.pi

# Antenna lengths
h1 = 0.25 * wavelength
h2 = 0.625 * wavelength

# Define the range of z values for both antenna lengths
z1 = np.linspace(-h1, h1, 1000)
z2 = np.linspace(-h2, h2, 1000)

# Define the current distribution function I(z) for both antenna lengths
def I(z, h, z1, z2, z3):
    return np.piecewise(z, [z > z2, z <= z2],
                        [lambda z: I_0 * np.sin(k * (z3 - z)) / np.sin(k * h),
                         lambda z: I_0 * np.sin(k * (z - z1)) / np.sin(k * h)])

# Calculate the current values using the piecewise function
current1 = I(z1, h1, -h1, 0, h1)
current2 = I(z2, h2, -h2, 0, h2)

# Plotting the results in a 2x1 subplot
fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Adjust figsize if needed

# Current distribution plot for h1
axs[0].plot(z1/h1, current1/I_0, label='$h = 0.25 \lambda$')
axs[0].set_xlabel('$\dfrac{z}{h}$')
axs[0].set_ylabel('$\dfrac{I(z)}{I_0}$', rotation='horizontal', labelpad=10, ha='right')
axs[0].set_title('Normalized Current Distribution for $h = 0.25\lambda$')
axs[0].legend()
axs[0].grid(True)

# Current distribution plot for h2
axs[1].plot(z2/h2, current2/I_0, label='$h = 0.625 \lambda$')
axs[1].set_xlabel('$\dfrac{z}{h}$')
axs[1].set_ylabel('$\dfrac{I(z)}{I_0}$', rotation='horizontal', labelpad=10, ha='right')
axs[1].set_title('Normalized Current Distribution for $h = 0.625\lambda$')
axs[1].legend()
axs[1].grid(True)

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()
