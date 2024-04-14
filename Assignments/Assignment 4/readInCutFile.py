import os
import pandas as pd

# Function to extract and process data blocks from a .cut file using pandas
def processDataBlocks(basePath, jobNumber, fileKey):
    # Create the path to the job folder
    #print(f'Job number: {jobNumber}')
    if int(jobNumber) < 10:
        jobFolder = f'Job_0{jobNumber}'
    else:
        jobFolder = f'Job_{jobNumber}'
    jobFolderPath = os.path.join(basePath, jobFolder)

    # Check if the folder exists
    if not os.path.exists(jobFolderPath):
        raise FileNotFoundError(f"Job folder not found: {jobFolderPath}")
    
    # Now find the right .cut file that has fileKey in its name
    cutFiles = [f for f in os.listdir(jobFolderPath) if fileKey in f]

    # Check if there is exactly one file with the key
    if len(cutFiles) != 1:
        raise FileNotFoundError(f"Expected exactly one file with key {fileKey} in {jobFolderPath}")
    
    # Get the path to the file
    filePath = os.path.join(jobFolderPath, cutFiles[0])

    # Check to make sure filePath exists
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"File not found: {filePath}")

    # Initialize an empty list to store the data blocks
    processedBlocks = []

    # Open the file and read the lines
    with open(filePath, 'r') as file:
        lines = file.readlines()

    # Variable to keep track of whether we are currently collecting data for a block
    collecting = False
    currentBlock = []

    # Process lines
    for line in lines:
        # Check if the line is a header or marker for a new block
        if "Field data in cuts" in line:
            # If we were already collecting a block, save it and start a new one
            if collecting and currentBlock:
                # Convert the current block to a DataFrame
                data_df = pd.DataFrame(currentBlock[1:], columns=['col1', 'col2', 'col3', 'col4'])
                # Create a dictionary for the metadata and the data DataFrame
                metadata = currentBlock[0]
                metadataDict = {
                    "vInit": metadata[0],
                    "vInc": metadata[1],
                    "vNum": int(metadata[2]),
                    "c": metadata[3],
                    "iComp": int(metadata[4]),
                    "iCut": int(metadata[5]),
                    "nComp": int(metadata[6])
                }
                processedBlocks.append((metadataDict, data_df))
                currentBlock = []
            collecting = True
        elif collecting:
            # If we are collecting data and the line is part of a data block, process the line
            currentBlock.append([float(x) for x in line.strip().split()])

    # Don't forget to add the last block if the file doesn't end with a header
    if currentBlock:
        data_df = pd.DataFrame(currentBlock[1:], columns=['col1', 'col2', 'col3', 'col4'])
        metadata = currentBlock[0]
        metadataDict = {
            "vInit": metadata[0],
            "vInc": metadata[1],
            "vNum": int(metadata[2]),
            "c": metadata[3],
            "iComp": int(metadata[4]),
            "iCut": int(metadata[5]),
            "nComp": int(metadata[6])
        }
        processedBlocks.append((metadataDict, data_df))

    return processedBlocks

# Function to print the processed job using pandas
def printProcessedJob(processedBlocks):
    for metadata, data_df in processedBlocks:
        print("\nDATA FROM GRASP JOB".center(40, "-"))
        metadata_df = pd.DataFrame([metadata])
        print(metadata_df.to_string(index=False))
        
        print("\nSIMULATED DATA FROM GRASP".center(40, "-"))
        print(data_df.to_string(index=False))

# basePath = 'Assignments\\Assignment 4\\QuestionOne'
# jobNumber = 19
# fileKey = 'single_cut.cut'
# filePath = os.path.join(basePath, f'Job_{jobNumber}', fileKey)

# # Process the data blocks in the file and print the first block
# processedBlocks = processDataBlocks(basePath=basePath, jobNumber=jobNumber, fileKey=fileKey)
# print(f'Processed {len(processedBlocks)} data blocks from {filePath}')
# printProcessedJob(processedBlocks[1:2])  
# #print(processedBlocks[2])
