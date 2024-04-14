'''
This generates plots for Question 1 of Assignment 4
Where a single dish is used, basic pattern cuts are shown and the effect of feed taper is shown for 2 different dishes
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
basePath = 'Assignments\\Assignment 4\\QuestionOne'
jobNumber = 19
fileKey = 'single_cut.cut'

# Ouestion 1 a
# Process the data blocks in the file and print the first block
# The data that is in the columns is the E field in components
# Col1 and Col2 are the co polarized E field components
# Col3 and Col4 are the cross polarized E field components
patternCuts = processDataBlocks(basePath=basePath, jobNumber=jobNumber, fileKey=fileKey)

# Plot the single 2D pattern cuts for a single sim (Q1a)
plotPatternCutsSingleJob(patternCuts)

# # Question 1 d
# # for the F/D = 0.4 case
# pltTitle = 'Normalized Radiation Intensity vs $\\theta$ for Various Tapers F/D = 0.4 (Deep Dish)'
# # Job numbers from grasp (for file names)
# jobStart = 20
# jobEnd = 46
# jobNumbers = range(jobStart, jobEnd + 1)

# # Amount of feed taper
# taperStart = -4 # in dB
# taperEnd = -30 # in dB
# taperStep = -1 # in dB
# taperValues = range(taperStart, taperEnd - 1, taperStep)

# # Check to make sure this is correct
# assert(len(taperValues) == len(jobNumbers))

# # Plot the radiation intensity for the F/D = 0.4 case
# # plotMultipleJobs(jobNumbers, taperValues, basePath, fileKey, plotTitle=pltTitle, normFactor=16.6, cbarTitle='Feed Taper (dB)')

# # for the F/D = 2 case
# # Job numbers from grasp (for file names)
# pltTitle = 'Normalized Radiation Intensity vs $\\theta$ for Various Tapers F/D = 2 (Shallow Dish)'
# jobStart = 47
# jobEnd = 73
# jobNumbers = range(jobStart, jobEnd + 1)

# # Amount of feed taper
# taperStart = -4 # in dB
# taperEnd = -30 # in dB
# taperStep = -1 # in dB
# taperValues = range(taperStart, taperEnd - 1, taperStep)

# # Check to make sure this is correct
# assert(len(taperValues) == len(jobNumbers))

# # Plot the radiation intensity for the F/D = 0.4 case
# plotMultipleJobs(jobNumbers, taperValues, basePath, fileKey, plotTitle=pltTitle, normFactor=0.02, cbarTitle='Feed Taper (dB)')






