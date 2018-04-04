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

import sys
import os


BASEDIR  = '/home/antonio/testSegmentation/'
CODEDIR  = os.path.join(BASEDIR, 'Code')
TESTSDIR = os.path.join(BASEDIR, 'Tests_LUVAR')

script_Training = os.path.join(CODEDIR, 'TrainingNetwork.py')

listModels     = ['Unet3D']
listOptimizers = ['Adam']
listLearnRates = ['1.0e-05', '3.0e-05', '1.0e-04', '3.0e-04', '1.0e-03', '3.0e-03', '1.0e-02', '3.0e-02']

for arg1 in listModels:
    for arg2 in listOptimizers:
        for arg3 in listLearnRates:

            os.system('python %s %s %s %s' %(script_Training, arg1, arg2, arg3))

            olddir = os.path.join(TESTSDIR, 'Models')
            newdir = os.path.join(TESTSDIR, 'Models_%s_%s_%s'%(arg1, arg2, arg3))

            os.system('mv %s %s'%(olddir, newdir))
            os.system('mkdir %s' % (olddir))