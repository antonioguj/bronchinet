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

from CommonUtil.FunctionsUtil import *
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
TESTSDIR   = joinpathnames(BASEDIR, 'Tests_LUVAR')
RESULTDIR  = joinpathnames(TESTSDIR, 'Models')
PREDICTDIR = joinpathnames(TESTSDIR, 'Predictions')
ATTRIBUTES = ['3DUnetShallow', 'size352x240x16', 'SlidingWindow', 'BatchGenerator', 'ConfineLungs']



# ********** LAUNCH TRAINING **********
measureTime = WallClockTime()

script_Training   = joinpathnames(CODEDIR, 'TrainingNetwork.py')

# Launching Training script
Popen_obj = subprocess.Popen(['python', script_Training])

# Wait for the process to finish
# I would like to implement a way to input signal to stop process
Popen_obj.wait()

print('<-Training performed in %s sec...->' %(measureTime.compute()))
# ********** LAUNCH TRAINING **********


# ********** MANAGE OUTPUT FILES **********
# remove all output files except: 'lossHistory.txt', and weights for last epoch, and minimum train and valid loss
listoutputfiles = listfilesDir(RESULTDIR)

listoutputfiles.remove('lossHistory.txt')

listfiles_epochs = []
listfiles_loss   = []
listfiles_valoss = []

for file in listoutputfiles:
    attributes = file.replace('model_','').replace('.hdf5','').split('_')
    listfiles_epochs.append(attributes[0])
    listfiles_loss  .append(attributes[1])
    listfiles_valoss.append(attributes[2])
#endfor

keepfile_maxepochs = listoutputfiles[listfiles_epochs.index(max(listfiles_epochs))]
keepfile_minloss   = listoutputfiles[listfiles_loss  .index(min(listfiles_loss  ))]
keepfile_minvaloss = listoutputfiles[listfiles_valoss.index(min(listfiles_valoss))]

# remove files
for file in listoutputfiles:
    if file not in [keepfile_maxepochs, keepfile_minloss, keepfile_minvaloss]:
        removefile(joinpathnames(RESULTDIR, file))
#endfor

# finally, link to rename remaining files
makelink(keepfile_maxepochs, joinpathnames(RESULTDIR, 'model_lastEpoch.hdf5'))
makelink(keepfile_minloss,   joinpathnames(RESULTDIR, 'model_minLoss.hdf5'))
makelink(keepfile_minvaloss, joinpathnames(RESULTDIR, 'model_minValoss.hdf5'))
# ********** MANAGE OUTPUT FILES **********


# ********** LAUNCH PREDICTION **********
measureTime = WallClockTime()

script_Prediction = joinpathnames(CODEDIR, 'PredictionModel.py')

# Launching Training script
Popen_obj = subprocess.Popen(['python', script_Prediction])

Popen_obj.wait()

print('<-Prediction performed in %s sec...->' %(measureTime.compute()))
# ********** LAUNCH PREDICTION **********


# ********** SAVE MODEL AND PREDICTIONS **********
LAST_EPOCH = max(listfiles_epochs)

newResultDir  = 'Models_%0.2i-%0.2i-%0.4i_'%(getdatetoday()) + '_'.join(ATTRIBUTES) + '_epoch%s'%(LAST_EPOCH)
newPredictDir = 'Predictions_%0.2i-%0.2i-%0.4i_'%(getdatetoday()) + '_'.join(ATTRIBUTES) + '_epoch%s'%(LAST_EPOCH)

if isExistdir(newResultDir):
    newResultDir = newResultDir + '_NEW'
if isExistdir(newPredictDir):
    newPredictDir = newPredictDir + '_NEW'

movedir(RESULTDIR,  joinpathnames(TESTSDIR, joinpathnames('ModelsSaved', newResultDir)))
movedir(PREDICTDIR, joinpathnames(TESTSDIR, joinpathnames('PredictionsSaved', newPredictDir)))
# ********** SAVE MODEL AND PREDICTIONS **********