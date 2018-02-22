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


BASEDIR = '/home/antonio/testSegmentation/Tests_LUVAR'

#IMAGES_HEIGHT = 331
IMAGES_HEIGHT = 512
#IMAGES_WIDTH  = 468
IMAGES_WIDTH  = 512

BOUNDBOX_CROPPING = [(99, 430),
                     (33, 501)]

FORMATIMAGEDATA       = np.int16
FORMATGROUNDTRUTHDATA = np.int8

MAXIMAGESTOLOAD = 100

SMOOTH = 1