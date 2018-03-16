#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

import numpy as np


BASEDIR = '/home/antonio/testSegmentation/Tests_LUVAR/'

# MUST BE MULTIPLES OF 16
IMAGES_HEIGHT = 288
#IMAGES_HEIGHT = 256
IMAGES_WIDTH  = 192
#IMAGES_WIDTH  = 256
IMAGES_DEPTHZ = 16

CROPPATCH = ((112, 400),
             (64,  256))

def getCROPPATCHMIRROR(cropPatch):
    return ((cropPatch[0][0],     cropPatch[0][1]),
            (512-cropPatch[1][1], 512-cropPatch[1][0]))

FORMATIMAGEDATA = np.int16
FORMATMASKDATA  = np.int8