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
import subprocess


BASEDIR  = '/home/antonio/testSegmentation/'
CODEDIR  = joinpathnames(BASEDIR, 'Code')
TESTSDIR = joinpathnames(BASEDIR, 'Tests_LUVAR')

script_Training = joinpathnames(CODEDIR, 'TrainingModel.py')

listModels     = ['Unet3D']
listOptimizers = ['Adam']
listLearnRates = ['1.0e-05', '3.0e-05', '1.0e-04', '3.0e-04', '1.0e-03', '3.0e-03', '1.0e-02', '3.0e-02']


for arg1 in listModels:
    for arg2 in listOptimizers:
        for arg3 in listLearnRates:

            measureTime = WallClockTime()

            # Launching Training script
            Popen_obj = subprocess.Popen(['python', script_Training, '--model', arg1, '--optimizer', arg2, '--learn_rate', arg3])

            # Wait for the process to finish
            # I would like to implement a way to input signal to stop process
            Popen_obj.wait()

            print('<-Training performed in %s sec...->' % (measureTime.compute()))

        #endfor
    #endfor
#endfor