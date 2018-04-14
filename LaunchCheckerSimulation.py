#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
#######################################################################################

#!/usr/bin/python

from CommonUtil.Constants import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.ResultsFilesManager import *
import subprocess

TIME_MAX_TRAINING = 1.0e+06
TIME_CHECK = 60

def PauseProgram():
    raw_input("<-Press the <ENTER> key to continue...->")

def CheckKillProgram():
    input = raw_input("<-Enter \"kill\" to stop the Training script...->")
    return (input=='kill')


BASEDIR    = '/home/antonio/testSegmentation/'
CODEDIR    = joinpathnames(BASEDIR, 'Code')
TESTSDIR   = joinpathnames(BASEDIR, 'Tests_LUVAR_LUNGS')
RESULTDIR  = joinpathnames(TESTSDIR, 'Models')
PREDICTDIR = joinpathnames(TESTSDIR, 'Predictions')


ATTRIBUTES = ['SegmentationLungs', '3DUnetShallow', 'size352x240x16', 'SlidingWindow']

# ARGUMENTS_TRAINING = {'--basedir' : TESTSDIR,
#                       '--model' : 'Unet3D_Shallow',
#                       '--optimizer' : 'Adam',
#                       '--lossfun' : 'BinaryCrossEntropy',
#                       '--metrics' : 'DiceCoefficient',
#                       '--use_dataAugmentation' : 'True',
#                       '--slidingWindowImages' : 'True',
#                       '--prop_overlap_Z_X_Y' : '0.5, 0.0, 0.0'}
#
# ARGUMENTS_TESTING = {'--basedir' : TESTSDIR,
#                      '--prediction_model' : 'last_Epoch',
#                      '--reconstructPrediction' : 'True',
#                      '--thresholdOutImages' : 'True',
#                      '--thresholdValue' : '0.5'}


ARGUMENTS_TRAINING = {}
ARGUMENTS_TESTING  = {}


# ********** LAUNCH TRAINING **********
# measureTime = WallClockTime()
#
# script_Training   = joinpathnames(CODEDIR, 'TrainingNetwork.py')
#
# # Launching Training script
# list_arguments = [i for (key, value) in ARGUMENTS_TRAINING.iteritems() for i in (key, value)]
#
# Popen_obj = subprocess.Popen(['python', script_Training] + list_arguments)
#
# # Wait for the process to finish
# # I would like to implement a way to input signal to stop process
# Popen_obj.wait()
#
# print('<-Training performed in %s sec...->' %(measureTime.compute()))
# ********** LAUNCH TRAINING **********



# ********** MANAGE OUTPUT FILES **********
resultsFilesManager = ResultsFilesManager(RESULTDIR)

if USE_RESTARTMODEL:
    resultsFilesManager.completeLossHistoryRestart()

resultsFilesManager.cleanUpResultsDir_EndTests()
# ********** MANAGE OUTPUT FILES **********



# ********** LAUNCH PREDICTION **********
# measureTime = WallClockTime()
#
# script_Prediction = joinpathnames(CODEDIR, 'PredictionModel.py')
#
# # Launching Training script
# list_arguments = [i for (key, value) in ARGUMENTS_TESTING.iteritems() for i in (key, value)]
#
# Popen_obj = subprocess.Popen(['python', script_Prediction] + list_arguments)
#
# Popen_obj.wait()
#
# print('<-Prediction performed in %s sec...->' %(measureTime.compute()))
# ********** LAUNCH PREDICTION **********



# ********** SAVE MODEL AND PREDICTIONS **********
# LAST_EPOCH = resultsFilesManager.computeLastEpochInFiles()
#
# newResultDir  = 'Models_%0.2i-%0.2i-%0.4i_'%(getdatetoday()) + '_'.join(ATTRIBUTES) + '_epoch%s'%(LAST_EPOCH)
# newPredictDir = 'Predictions_%0.2i-%0.2i-%0.4i_'%(getdatetoday()) + '_'.join(ATTRIBUTES) + '_epoch%s'%(LAST_EPOCH)
#
# if isExistdir(newResultDir):
#     newResultDir = newResultDir + '_NEW'
# if isExistdir(newPredictDir):
#     newPredictDir = newPredictDir + '_NEW'
#
# movedir(RESULTDIR,  joinpathnames(TESTSDIR, joinpathnames('ModelsSaved', newResultDir)))
# movedir(PREDICTDIR, joinpathnames(TESTSDIR, joinpathnames('PredictionsSaved', newPredictDir)))
# ********** SAVE MODEL AND PREDICTIONS **********