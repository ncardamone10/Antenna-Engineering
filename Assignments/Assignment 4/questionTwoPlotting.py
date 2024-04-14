'''
This generates plots for Question 2 of Assignment 4
Where a single dish is used and the effect of axial defocusing is shown
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.integrate import quad
from scipy.integrate import simpson
import matplotlib.cm as cm

from readInCutFile import processDataBlocks, printProcessedJob
from plottingPatternCuts import plotPatternCutsSingleJob, plotMultipleJobs


# Paths to question one data
basePath = 'Assignments\\Assignment 4\\QuestionTwo'
fileKey = 'single_cut.cut'


pltTitle = 'Normalized Radiation Intensity vs $\\theta$ for Various Amounts of Axial Defocusing'
cBarTitle = 'Axial Defocusing ($\\lambda$)'
# Job numbers from grasp (for file names)
jobStart = 48-23#2
jobEnd = 48
jobNumbers = range(jobStart, jobEnd + 1)

# Amount of feed taper
wavelength = 15 # in mm
defocusStart = 0#-23 # in mm
defocusEnd = 23 # in mm
defocusStep = 1 # in mm
defocusValues = range(defocusStart, defocusEnd + 1, defocusStep)

# divide each defocus value by the wavelength
defocusValues = [defocusValue / wavelength for defocusValue in defocusValues]

print(f'Number of defocus values: {len(defocusValues)}')
print(f'Number of job numbers: {len(jobNumbers)}')
# Check to make sure this is correct
assert(len(defocusValues) == len(jobNumbers))

# Plot the radiation intensity for the F/D = 0.4 case
plotMultipleJobs(jobNumbers, defocusValues, basePath, fileKey, plotTitle=pltTitle, normFactor=16.6, cbarTitle=cBarTitle)





