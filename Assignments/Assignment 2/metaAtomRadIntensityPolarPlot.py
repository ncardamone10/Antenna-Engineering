import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Plotting Constants
phi = [0, 45, 90]
EPSILON = 1e-20

# Plotting Variables
numberOfPoints = 100
theta = np.linspace(0, np.pi, numberOfPoints)

# Free Space Wave Impedance
eta = 120*np.pi

# Radiation Intensity
def Emag3(theta, phi, EPSILON=1e-20):
    theta = theta + EPSILON   # Small constant to avoid division by zero
    phi = phi + EPSILON   # Small constant to avoid division by zero
    Etheta = np.sin(phi) - np.cos(theta)*np.sin(phi)
    Etheta = Etheta/(4*np.pi)
    Ephi = np.cos(theta)*np.cos(phi) - np.cos(phi)
    Ephi = Ephi/(4*np.pi)
    
    # Matrix to convert from rect to spherical
    rectToSph = np.array([
        [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),  np.cos(theta)],
        [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
        [-np.sin(phi),              np.cos(phi),                0            ]
    ])
    
    
    EvecSph = [0, Etheta, Ephi]
    EvecCart = np.linalg.inv(rectToSph) @ EvecSph
    Emag = np.sqrt(EvecCart[0]**2 + EvecCart[1]**2 + EvecCart[2]**2)
    return Emag

def EmagDipole(theta, phi, EPSILON=1e-20, kh=0.5):
    theta = theta + EPSILON   # Small constant to avoid division by zero
    phi = phi + EPSILON   # Small constant to avoid division by zero
    
    Etheta = (np.cos(kh*np.cos(theta)) - np.cos(kh))/(np.sin(theta))
    Ephi = 0
    
    # Matrix to convert from rect to spherical
    rectToSph = np.array([
        [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),  np.cos(theta)],
        [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
        [-np.sin(phi),              np.cos(phi),                0            ]
    ])
    
    
    EvecSph = [0, Etheta, Ephi]
    EvecCart = np.linalg.inv(rectToSph) @ EvecSph
    Emag = np.sqrt(EvecCart[0]**2 + EvecCart[1]**2 + EvecCart[2]**2)
    return Emag


def mirror_data(theta, data):
    # Flip the arrays
    theta_flipped = np.flip(theta)
    data_flipped = np.flip(data)

    # Subtract the flipped theta values from 180 to mirror across theta = 90
    theta_flipped = - theta_flipped

    # Concatenate the original and flipped arrays
    theta_mirrored = np.concatenate((theta_flipped, theta))
    data_mirrored = np.concatenate((data, data_flipped))

    return theta_mirrored, data_mirrored


# Loop through the phi0 values and plot the normalized Co and Cross Polarizations on polar plots
#fig, axs = plt.subplots(2, 2)  # Create a 2x2 subplot

# Create a new figure
fig = plt.figure()

# Create a GridSpec object
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0], polar=True, frame_on=False)
ax3 = fig.add_subplot(gs[2], polar=True, frame_on=False)

ax2 = fig.add_subplot(gs[1])
ax4 = fig.add_subplot(gs[3])

for i in range(len(phi)):
    EmagVec = []
    for theta0 in theta:
        EmagVec.append(np.abs(Emag3(theta0, phi[i]*np.pi/180)))
        
    if np.max(EmagVec) == 0:
        print(f'$Error: Emag phi = {phi[i]}^\circ is zero$')
    else:     
        EmagVec = EmagVec/(np.max(EmagVec))  # Normalize the E field to 1 for plotting



    theta_mirrored, Emirrored = mirror_data(theta, EmagVec)
    
    ax1.plot(theta_mirrored, 20*np.log10(Emirrored + EPSILON), label=f'$\phi = {phi[i]} degrees$')
    ax2.plot(theta_mirrored/np.pi*180, 20*np.log10(Emirrored + EPSILON), label=f'$\phi = {phi[i]} degrees$')
    ax3.plot(theta_mirrored, (Emirrored + EPSILON), label=f'$\phi = {phi[i]} degrees$')
    ax4.plot(theta_mirrored/np.pi*180, (Emirrored + EPSILON), label=f'$\phi = {phi[i]} degrees$')



    
ax1.set_rlim(-100, 0)  
ax2.set_ylim(-100, 0)  
ax4.set_ylim(0, 1)  

ax1.set_title('Meta Atom Radiation Intensity (dBi)')
ax2.set_title('Meta Atom Radiation Intensity (dBi)')
ax3.set_title('Meta Atom Radiation Intensity (Linear Scale)')
ax4.set_title('Meta Atom Radiation Intensity (Linear Scale)')

ax1.set_theta_offset(np.pi/2) 
ax1.set_theta_direction(-1) 
ax3.set_theta_offset(np.pi/2) 
ax3.set_theta_direction(-1) 

#ax1.legend(loc='center left')
ax2.legend(loc='lower center') 
ax3.legend(loc='lower center')   
ax4.legend(loc='lower center')  

ax2.grid(True)
ax4.grid(True)

ax2.set_xlabel(r'$\theta$ (degrees)')
ax2.set_ylabel(r'$U$ (dBi)')
ax4.set_xlabel(r'$\theta$ (degrees)')
ax4.set_ylabel(r'$U$ ')


plt.tight_layout()  # To prevent overlapping
plt.show()





