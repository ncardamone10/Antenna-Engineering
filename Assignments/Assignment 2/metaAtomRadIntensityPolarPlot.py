import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Plotting Constants
phi = [0, 90, 180, 270]
EPSILON = 1e-20

# Plotting Variables
numberOfPoints = 100
theta = np.linspace(0, np.pi/2, numberOfPoints)

# Free Space Wave Impedance
eta = 120*np.pi

# Radiation Intensity
def U(theta, phi, EPSILON=1e-20):
    theta = theta + EPSILON   # Small constant to avoid division by zero
    phi = phi + EPSILON   # Small constant to avoid division by zero
    terms = np.zeros(7)
    terms[0] = 1/np.tan(theta)**2
    terms[1] = -4*np.tan(phi)/(np.cos(theta)*np.tan(theta)**2)
    terms[2] = np.tan(phi)**2
    terms[3] = 1/np.sin(theta)**2
    terms[4] = -1
    terms[5] = np.tan(phi)**2/np.tan(theta)**2
    terms[6] = 1/(np.tan(theta)*np.sin(phi))**2
    result = np.sum(terms)/(32*np.pi**2*eta)
    return result

def Emag(theta, phi, EPSILON=1e-20):
    theta = theta + EPSILON   # Small constant to avoid division by zero
    phi = phi + EPSILON   # Small constant to avoid division by zero
    Etheta = np.tan(phi)/np.sin(theta) - 1/np.tan(theta)
    Etheta = Etheta/(4*np.pi)
    Ephi = 1/np.tan(theta) - 1/(np.sin(theta)*np.tan(phi))
    Ephi = Ephi/(4*np.pi)
    Emag = np.sqrt(Etheta**2 + Ephi**2)
    return Emag






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


# Loop through the phi0 values and plot the normalized Co and Cross Polarizations on polar plots
#fig, axs = plt.subplots(2, 2)  # Create a 2x2 subplot

# Create a new figure
fig = plt.figure()

# Create a GridSpec object
gs = GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0], polar=True, frame_on=False)
ax3 = fig.add_subplot(gs[1])


for i in range(len(phi)):
    # radiationIntensity = []
    # for theta0 in theta:
    #     radiationIntensity.append(np.abs(U(theta0, phi[i]*np.pi/180)))
        
    # if np.max(radiationIntensity) == 0:
    #     print(f'$Error: Co Pol phi = {phi[i]}^\circ is zero$')
    # else:     
    #     radiationIntensity = radiationIntensity/(np.max(radiationIntensity))  # Normalize the E field to 1 for plotting

    EmagVec = []
    for theta0 in theta:
        EmagVec.append(np.abs(Emag(theta0, phi[i]*np.pi/180)))
        
    if np.max(EmagVec) == 0:
        print(f'$Error: Emag phi = {phi[i]}^\circ is zero$')
    else:     
        EmagVec = EmagVec/(np.max(EmagVec))  # Normalize the E field to 1 for plotting



    theta_mirrored, Emirrored = mirror_data(theta, EmagVec)
    
    ax1.plot(theta_mirrored, 20*np.log10(Emirrored + EPSILON), label=f'$\phi = {phi[i]}^\circ$')
    ax3.plot(theta_mirrored/np.pi*180, 20*np.log10(Emirrored + EPSILON), label=f'$\phi = {phi[i]}^\circ$')

      


    
#ax1.set_rlim(-100, 0)  
ax3.set_ylim(-100, 0)  
ax3.set_xlim(-1, 1)

ax1.set_title('Meta Atom Radiation Intensity (dBi)')
ax3.set_title('Meta Atom Radiation Intensity (dBi)')


ax1.set_theta_offset(np.pi/2) 
ax1.set_theta_direction(-1) 


ax1.legend(loc='lower center')   
ax3.legend(loc='lower left')   

ax3.grid(True)


ax3.set_xlabel(r'$\theta$ (degrees)')
ax3.set_ylabel(r'$U$ (dBi)')



plt.tight_layout()  # To prevent overlapping
plt.show()





