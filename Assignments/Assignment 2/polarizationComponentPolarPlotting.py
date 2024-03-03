# Plot the far field (E field) pf co and cross polarizations on polar plots


import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Plotting Constants
qe = 3.2
qh = 2.16
phi0 = [0, 45, 90]


# Plotting Variables
numberOfPoints = 100
theta = np.linspace(0, np.pi/2, numberOfPoints)


# Far Field (E Field) Co Polarization
def E_co(theta, phi, qe, qh):
    result = np.abs(np.cos(theta))**(2*qe)*np.sin(phi)**2 + np.abs(np.cos(theta))**(2*qh)*np.cos(phi)**2
    return result #np.where((0 <= theta) & (theta <= np.pi/2), result, EPSILON)

# Far Field (E Field) Cross Polarization
def E_cross(theta, phi, qe, qh):
    #print(f'$Ecross: phi = {phi}, sin = {np.sin(2*phi)}$')
    result = ((np.cos(theta))**(2*qe) - (np.cos(theta))**(2*qh))*np.sin(2*phi)/2
    return result #np.where((0 <= theta) & (theta <= np.pi/2), result, EPSILON)

def mirror_data(theta, data):
    # Flip the arrays
    theta_flipped = np.flip(theta)
    data_flipped = np.flip(data)

    # Subtract the flipped theta values from 180 to mirror across theta = 90
    theta_flipped = - theta_flipped

    # Concatenate the original and flipped arrays
    theta_mirrored = np.concatenate((theta, theta_flipped))
    data_mirrored = np.concatenate((data, data_flipped))

    return theta_mirrored, data_mirrored

EPSILON = 1e-20  # Small constant to avoid division by zero
# Loop through the phi0 values and plot the normalized Co and Cross Polarizations on polar plots
#fig, axs = plt.subplots(2, 2)  # Create a 2x2 subplot

# Create a new figure
fig = plt.figure()

# Create a GridSpec object
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0], polar=True, frame_on=False)
ax2 = fig.add_subplot(gs[0, 1], polar=True, frame_on=False)
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# axs[0, 0] = fig.add_subplot(2, 2, 1, polar=True, frame_on=False)  # First subplot for Co Polarization  # First subplot for Co Polarization
# axs[0, 1] = plt.subplot(2, 2, 2, polar=True)  # Second subplot for Cross Polarization




for i in range(len(phi0)):
    Eco = []
    Ecr = []
    for theta0 in theta:
        Eco.append(np.abs(E_co(theta0, phi0[i]*np.pi/180, qe, qh)))
        Ecr.append(np.abs(E_cross(theta0, phi0[i]*np.pi/180, qe, qh)))
    if np.max(Eco) == 0:
        print(f'$Error: Co Pol phi = {phi0[i]}^\circ is zero$')
    else:     
        Eco = Eco/(np.max(Eco))  # Normalize the E field to 1 for plotting
    if np.max(Ecr) == 0:
        print(f'$Error: Cross Pol phi = {phi0[i]}^\circ is zero$')
    else:   
        Ecr = Ecr/(np.max(Ecr))  # Normalize the E field to 1 for plotting
    #Ecr = np.maximum(Ecr, EPSILON) 
    #Eco = np.maximum(Eco, EPSILON) 
    
    try: 
        theta_mirrored, Eco_mirrored = mirror_data(theta, Eco)
        
        ax1.plot(theta_mirrored, 10*np.log10(Eco_mirrored + EPSILON), label=f'$\phi = {phi0[i]}^\circ$')
        ax3.plot(theta_mirrored/np.pi*180, 10*np.log10(Eco_mirrored + EPSILON), label=f'$\phi = {phi0[i]}^\circ$')

    except Exception as e:
        print(f'$Error: can\'t plot for Co Pol phi = {phi0[i]}^\circ$')
        ax1.plot(theta, 10*np.log10(EPSILON*np.ones_like(theta)), label=f'$\phi = {phi0[i]}^\circ$')
        ax3.plot(theta/np.pi*180, 10*np.log10(EPSILON*np.ones_like(theta)), label=f'$\phi = {phi0[i]}^\circ$')
        
    
    if i == 1:
        theta_mirrored, Ecr_mirrored = mirror_data(theta, Ecr)
        ax2.plot(theta_mirrored, 10*np.log10(Ecr_mirrored + EPSILON), label=f'$\phi = {phi0[i]}^\circ$')
        ax4.plot(theta_mirrored/np.pi*180, 10*np.log10(Ecr_mirrored + EPSILON), label=f'$\phi = {phi0[i]}^\circ$')
    else:
        print(f'$Error: can\'t plot for Cross Pol phi = {phi0[i]}^\circ$')
        ax2.plot(theta, 10*np.log10(EPSILON*np.ones_like(theta)), label=f'$\phi = {phi0[i]}^\circ$')
        ax4.plot(theta/np.pi*180, 10*np.log10(EPSILON*np.ones_like(theta)), label=f'$\phi = {phi0[i]}^\circ$')

    
ax1.set_rlim(-40, 0)  # Set the limits for the Co Polarization
ax2.set_rlim(-40, 0)  # Set the limits for the Cross Polarization    
ax3.set_ylim(-40, 0)  # Set the limits for the Co Polarization
ax4.set_ylim(-40, 0)  # Set the limits for the Co Polarization

ax1.set_title('Co Polarization (dBr)')
ax2.set_title('Cross Polarization (dBr)')
ax3.set_title('Co Polarization (dBr)')
ax4.set_title('Cross Polarization (dBr)')

ax1.set_theta_offset(np.pi/2) 
ax2.set_theta_offset(np.pi/2) 

ax1.set_theta_direction(-1) 
ax2.set_theta_direction(-1) 

ax1.legend(loc='lower center')   
ax2.legend(loc='lower center')
ax3.legend(loc='lower left')   
ax4.legend(loc='lower left')

ax3.grid(True)
ax4.grid(True)

ax3.set_xlabel(r'$\theta$ (degrees)')
ax3.set_ylabel(r'$E_{copol}$ (dBr)')

ax4.set_xlabel(r'$\theta$ (degrees)')   
ax4.set_ylabel(r'$E_{crosspol}$ (dBr)')


plt.tight_layout()  # To prevent overlapping
plt.show()
    










