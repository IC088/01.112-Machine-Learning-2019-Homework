import os
import sys
import math
import copy
import json
import argparse

#easier access when need to get the log value
def log(val):
    if val == 0:
        return - sys.maxint - 1
    return math.log(val)

# Process Data

def process_data( filePath ):
    tags = {}  # save the number of times each tag appears
    observations = {}  # save the word observation
    labelled_observations = {}  # save labelled observations
    
    for line in open(filePath, 'r', encoding="utf8"):
        segmentedLine = line.rstrip()
        if segmentedLine:  # if its not just an empty string
            segmentedLine = segmentedLine.rsplit(' ', 1)

            observation = segmentedLine[0]  # X
            tag = segmentedLine[1]  # Y

            if observation not in observations:  # if this observation has never been seen before
                observations[observation] = 1
            else:  # if this observation has been seen before
                observations[observation] += 1

            if tag not in tags:  # if this tag has never been seen before
                tags[tag] = 1
                labelled_observations[tag] = {observation: 1}

            else:  # if this tag has been seen before
                tags[tag] += 1
                if observation not in labelled_observations[tag]:
                    labelled_observations[tag][observation] = 1
                else:
                    labelled_observations[tag][observation] += 1
    return observations, tags, labelled_observations

# Estimate the emission probability
# Check whether it is unkown or not

def estimateEmission(observations, tags, labelled_observations , k=3): # As speciffied in the requirements k = 3   
    estimates = {} # save the estimates for each 
    for tag in labelled_observations:
        estimates[tag] = {}
        labelled_observations[tag]['##UNK##'] = 0
        for observation in list(labelled_observations[tag]):  # loop over all keys in labelled_observations
            if observation == '##UNK##': 
                continue
            if observation not in observations:  # if this observation has been found to appear less than k times before
                labelled_observations[tag]['##UNK##'] += labelled_observations[tag].pop(observation)
            elif observations[observation] < k:  # if first meet an observation that appear less than k times
                labelled_observations[tag]['##UNK##'] += labelled_observations[tag].pop(observation)
                del observations[observation]
            else:  # compute the MLE for that emission
                estimates[tag][observation] = float(labelled_observations[tag][observation]) / tags[tag] ## MLE
        estimates[tag]['##UNK##'] = float(labelled_observations[tag]['##UNK##']) / tags[tag]

    return list(observations), estimates

# Create the sentiment analysis 
def sentimentAnalysis(inputPath, estimates, outputPath):
    f = open(outputPath, 'w', encoding="utf8") # specified to be written as dev.p2.out
    unkPrediction = ('##UNK##', 0.0)
    for tag in estimates:
        if estimates[tag]['##UNK##'] > unkPrediction[1]:
            unkPrediction = (tag, estimates[tag]['##UNK##'])
    for line in open(inputPath, 'r', encoding="utf8"):
        observation = line.rstrip()
        if observation:
            prediction = ('', 0.0)  # prediction is tuple of tag and the MLE of observation for the given tag
            for tag in estimates:
                if observation in estimates[tag] and estimates[tag][observation] > prediction[1]:
                    prediction = (tag, estimates[tag][observation])
            if prediction[0]:
                f.write('%s %s\n' % (observation, prediction[0]))
            else:
                f.write('%s %s\n' % (observation, unkPrediction[0]))
        else:
            f.write('\n')

    print('Finished writing to file %s' % (outputPath))
    return f.close()    

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, dest='dataset', help='Dataset to run script over', required=True)


## Parsing the Dataset

args = parser.parse_args()

trainFilePath = f'./Dataset/{str(args.dataset)}/{str(args.dataset)}/train'
inputTestFilePath = f'./Dataset/{str(args.dataset)}/{str(args.dataset)}/dev.in'
outputTestFilePath = f'./Dataset/{str(args.dataset)}/{str(args.dataset)}/dev.p2.out'

observations, tags, labelled_observations = process_data( str(trainFilePath) )

m_training, estimates = estimateEmission(observations, tags, labelled_observations)
sentimentAnalysis(inputTestFilePath, estimates, outputTestFilePath)
