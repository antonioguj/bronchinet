#!/usr/bin/python

import sys
import os


nameScript = 'Code/TrainingUnet3D.py'

DIRTESTS = '/home/antonio/testSegmentation/Tests_LUVAR/'

listModels     = ['Unet3D']
listOptimizers = ['Adam']
listLearnRates = ['1.0e-05', '3.0e-05', '1.0e-04', '3.0e-04', '1.0e-03', '3.0e-03', '1.0e-02', '3.0e-02']

for arg1 in listModels:
    for arg2 in listOptimizers:
        for arg3 in listLearnRates:

            os.system('python %s %s %s %s' %(nameScript, arg1, arg2, arg3))

            olddir = os.path.join(DIRTESTS, 'Models')
            newdir = os.path.join(DIRTESTS, 'Models_%s_%s_%s'%(arg1, arg2, arg3))

            os.system('mv %s %s'%(olddir, newdir))
            os.system('mkdir %s' % (olddir))