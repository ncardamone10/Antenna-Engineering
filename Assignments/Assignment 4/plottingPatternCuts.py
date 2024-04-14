import os
import pandas as pd
import matplotlib.pyplot as plt
from readInCutFile import processDataBlocks, printProcessedJob
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.integrate import quad
from scipy.integrate import simpson
import matplotlib.cm as cm


# Function to plot the radiation intensity values
def plotPatternCutsSingleJob(patternCuts):
    maxUco, maxUcr = 0, 0  # Initialize maximum values

    # Collect all radiation intensity values to find the maximum
    allUco, allUcr = [], []
    theta_values = []
    phiValues = []

    for metadata, data_df in patternCuts:
        vInit = metadata['vInit']
        vInc = metadata['vInc']
        vNum = metadata['vNum']
        theta = np.linspace(vInit, vInit + vInc * (vNum - 1), vNum)
        phiValues.append(metadata['c'])
        #print(f'Phi: {metadata["c"]:.4f}°')

        # Store theta values for plotting
        theta_values.append(theta)

        col1 = data_df.iloc[:, 0]
        col2 = data_df.iloc[:, 1]
        col3 = data_df.iloc[:, 2]
        col4 = data_df.iloc[:, 3]

        Eco = np.sqrt(col1**2 + col2**2)
        Ecr = np.sqrt(col3**2 + col4**2)

        eta = 120 * np.pi  # Wave impedance
        Uco = np.abs(Eco)**2 / (2 * eta)
        Ucr = np.abs(Ecr)**2 / (2 * eta)

        # Replace NaN values and add a small epsilon to avoid log(0)
        EPSILON = 1e-40
        Uco = np.where(np.isnan(Uco), EPSILON, Uco)
        Ucr = np.where(np.isnan(Ucr), EPSILON, Ucr)

        allUco.append(Uco)
        allUcr.append(Ucr)

        # Update maximum values
        maxUco = max(maxUco, np.max(Uco))
        maxUcr = max(maxUcr, np.max(Ucr))

    # Plotting the radiation intensity values
    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Normalized Radiation Intensity vs $\\theta$')

    titles = ['Radiation Intensity (Co Pol)', 'Radiation Intensity (Cross Pol)']
    allU = [allUco, allUcr]

    maxU = np.max(allU)

    for i, ax in enumerate(axs):
        phiCount = 0
        for theta, U in zip(theta_values, allU[i]):
            ax.plot(theta, 10 * np.log10(U/maxU + EPSILON), label=f'$\\phi$={phiValues[phiCount]:.0f}°')
            phiCount += 1
        ax.set_ylabel('Normalized Radiation Intensity (dBr)')
        ax.set_xlabel('$\\theta$')
        ax.set_title(titles[i])
        ax.legend()
        ax.grid()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}°'))

    plt.tight_layout()
    plt.show()

# Now go through each job and plot the normalized radiation intensity
# This should give a 2x3 subplot where the top subplots are the co polarized radiation intensity and the bottom subplots are the cross polarized radiation intensity
# Each column should be a different value of phi (0, 45, 90)
# The label for each curve should be the value of the feed swParam
def plotMultipleJobs(jobNumbers, sweepParams, basePath, fileKey, plotTitle='', normFactor=1, cbarTitle=''):
    # Prepare the subplots: 2 rows for co-polarized and cross-polarized, 3 columns for phi values
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(plotTitle)

    phi_values = [0, 45, 90]  # Define the phi values to plot in each column

    # Initialize max values for normalization
    max_values = np.zeros((2, 3))  # For storing max values for normalization

    # Data collection for plotting
    all_data = {phi: {'co': [], 'cross': []} for phi in phi_values}

    for jobNumber, swParam in zip(jobNumbers, sweepParams):
        patternCuts = processDataBlocks(basePath=basePath, jobNumber=jobNumber, fileKey=fileKey)

        for metadata, data_df in patternCuts:
            phi = metadata['c']
            if phi not in phi_values:
                continue

            theta = np.linspace(metadata['vInit'], metadata['vInit'] + metadata['vInc'] * (metadata['vNum'] - 1), metadata['vNum'])
            col1 = data_df.iloc[:, 0]
            col2 = data_df.iloc[:, 1]
            col3 = data_df.iloc[:, 2]
            col4 = data_df.iloc[:, 3]

            Eco = np.sqrt(col1**2 + col2**2)
            Ecr = np.sqrt(col3**2 + col4**2)

            eta = 120 * np.pi  # Wave impedance
            Uco = np.abs(Eco)**2 / (2 * eta)
            Ucr = np.abs(Ecr)**2 / (2 * eta)

            EPSILON = 1e-40
            Uco = np.where(np.isnan(Uco), EPSILON, Uco)
            Ucr = np.where(np.isnan(Ucr), EPSILON, Ucr)

            col_index = phi_values.index(phi)
            max_values[0, col_index] = max(max_values[0, col_index], np.max(Uco))
            max_values[1, col_index] = max(max_values[1, col_index], np.max(Ucr))

            all_data[phi]['co'].append((theta, Uco, swParam))
            all_data[phi]['cross'].append((theta, Ucr, swParam))

    # Color bar setup
    cmap = cm.rainbow
    norm = plt.Normalize(min(sweepParams), max(sweepParams))

    # Plotting
    for col_index, phi in enumerate(phi_values):
        for row_index, pol in enumerate(['co', 'cross']):
            ax = axs[row_index, col_index]
            data = all_data[phi][pol]
            for (theta, U, swParam) in data:
                normalized_U = U /normFactor#/ max_values[row_index, col_index] + EPSILON
                sc = ax.plot(theta, 10 * np.log10(normalized_U), color=cmap(norm(swParam)))

            ax.set_title(f'$\\phi$ = {phi}° U {pol.title()} Pol')
            ax.set_xlabel('$\\theta$')
            ax.set_ylabel('Normalized Radiation Intensity (dBr)')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}°'))
            ax.grid()

            # Add colorbar to each subplot
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', aspect=10, shrink=0.6, pad=0.05)
            cbar.set_label(cbarTitle)

    plt.tight_layout()
    plt.show()



