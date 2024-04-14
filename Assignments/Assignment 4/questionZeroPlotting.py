import os
import pandas as pd
import matplotlib.pyplot as plt
from readInCutFile import processDataBlocks, printProcessedJob
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.integrate import quad
from scipy.integrate import simpson

# Paths to question one data
basePath = 'Assignments\\Assignment 4\\QuestionOne'
jobNumber = 79
fileKey = 'single_cut.cut'

# Process the data blocks in the file and print the first block
# The data that is in the columns is the E field in components
# Col1 and Col2 are the co polarized E field components
# Col3 and Col4 are the cross polarized E field components
patternCuts = processDataBlocks(basePath=basePath, jobNumber=jobNumber, fileKey=fileKey)


def plotPatternCuts(patternCuts):
    # Create 2x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('E Field Components, Radiation Intensity, and Directivity values vs $\\theta$')

    for index, (metadata, data_df) in enumerate(patternCuts):
        # Generate the degree values for the x-axis from the metadata
        vInit = metadata['vInit']
        vInc = metadata['vInc']
        vNum = metadata['vNum']
        theta = np.linspace(vInit, vInit + vInc * (vNum - 1), vNum)

        # Get the phi value from the metadata
        phi = metadata['c']


        # Getting the field data for the current block
        col1 = data_df.iloc[:, 0]
        col2 = data_df.iloc[:, 1]
        col3 = data_df.iloc[:, 2]
        col4 = data_df.iloc[:, 3]

        # Calculate the E fields
        Eco = np.sqrt(col1**2 + col2**2)
        Ecr = np.sqrt(col3**2 + col4**2)

        # Wave impedance
        eta = 120 * np.pi

        # Radiation Intensity
        Uco = np.abs(Eco)**2 / (2 * eta)
        Ucr = np.abs(Ecr)**2 / (2 * eta)

        # Directivity
        sinTheta = np.sin(np.deg2rad(theta))
        # 2 is there because I'm simulating from theta = 0 to theta = 180
        # 2 should be removed otherwise as data is symmetric
        # UcoAvg = 2*np.average(Uco, weights=sinTheta)
        # UcrAvg = 2*np.average(Ucr, weights=sinTheta)
        # Dco = Uco / UcoAvg
        # Dcr = Ucr / UcrAvg
        # Calculate the integral of Uco*sinTheta and Ucr*sinTheta
        # theta needs to be symetric and all the way around for this to work
        UcoIntegral = simpson(Uco * sinTheta, np.deg2rad(theta))
        UcrIntegral = simpson(Ucr * sinTheta, np.deg2rad(theta))

        print(f'Index: {index}, UcoIntegral: {UcoIntegral}, UcrIntegral: {UcrIntegral}')
        print(f'Drc')
        
        # Calculate the average U values
        UcoAvg = 0.5 * UcoIntegral
        UcrAvg = 0.5 * UcrIntegral
        
        # Calculate Dco and Dcr
        Dco = Uco / UcoAvg
        Dcr = Ucr / UcrAvg
        print(f'Dcr: {Dcr}')
        
        EPSILON = 1e-40

        # Replace NaN values in Eco, Ecr, Uco, Ucr, Dco, Dcr
        Eco = np.where(np.isnan(Eco), EPSILON, Eco)
        Ecr = np.where(np.isnan(Ecr), EPSILON, Ecr)
        Uco = np.where(np.isnan(Uco), EPSILON, Uco)
        Ucr = np.where(np.isnan(Ucr), EPSILON, Ucr)
        Dco = np.where(np.isnan(Dco), EPSILON, Dco)
        Dcr = np.where(np.isnan(Dcr), EPSILON, Dcr)

# Continue with your plotting
        # Plotting for each cut
        axs[0, 0].plot(theta, 20*np.log10(np.abs(Eco) + EPSILON), label=f'$E_{{co}}$, $\\phi$={phi:.0f}$\\degree$')
        axs[1, 0].plot(theta, 20*np.log10(np.abs(Ecr) + EPSILON), label=f'$E_{{cr}}$, $\\phi$={phi:.0f}$\\degree$')
        axs[0, 1].plot(theta, 10*np.log10(Uco + EPSILON), label=f'$U_{{co}}$, $\\phi$={phi:.0f}$\\degree$')
        axs[1, 1].plot(theta, 10*np.log10(Ucr + EPSILON), label=f'$U_{{cr}}$, $\\phi$={phi:.0f}$\\degree$')
        axs[0, 2].plot(theta, 10*np.log10(Dco + EPSILON), label=f'$D_{{co}}$, $\\phi$={phi:.0f}$\\degree$')
        axs[1, 2].plot(theta, 10*np.log10(Dcr + EPSILON), label=f'$D_{{cr}}$, $\\phi$={phi:.0f}$\\degree$')

    # Set labels for all subplots
    axs[0, 0].set_ylabel('E Field (dBV/m)')
    axs[1, 0].set_ylabel('E Field (dBV/m)')
    axs[0, 1].set_ylabel('Radiation Intensity (dBW/sr)')
    axs[1, 1].set_ylabel('Radiation Intensity (dBW/sr)')
    axs[0, 2].set_ylabel('Directivity (dBi)')
    axs[1, 2].set_ylabel('Directivity (dBi)')


    # Set x labels and grids for all subplots
    for ax_row in axs:
        for ax in ax_row:
            ax.set_xlabel('$\\theta$')
            ax.legend()
            ax.grid()
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}$\\degree$'))
    # Set titles for all subplots
    titles = [
    ['Co Pol E Field', 'Radiation Intensity (Co Pol)', 'Directivity (Co Pol)'],
    ['Cross Pol E Field', 'Radiation Intensity (Cross Pol)', 'Directivity (Cross Pol)']
]

    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            ax.set_title(titles[i][j])
    plt.tight_layout()
    plt.show()


# Now call the function with the pattern cuts
plotPatternCuts(patternCuts)


