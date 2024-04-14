'''
This generates plots for Question 5 of Assignment 4
Where a single hybolic dish is used and the pattern is shown
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
basePath = 'Assignments\\Assignment 4\\QuestionFive'
jobNumber = 5
fileKey = 'single_cut.cut'


# Process the data blocks in the file and print the first block
# The data that is in the columns is the E field in components
# Col1 and Col2 are the co polarized E field components
# Col3 and Col4 are the cross polarized E field components
patternCuts = processDataBlocks(basePath=basePath, jobNumber=jobNumber, fileKey=fileKey)

# Plot the single 2D pattern cuts for a single sim (Q1a)
plotPatternCutsSingleJob(patternCuts)
