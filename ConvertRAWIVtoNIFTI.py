#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from matplotlib import pyplot as plt
from StringIO import StringIO
from PIL import Image
import imageio
import numpy as np
import rawpy
from glob import glob
import os


# LOADING DATA
# ----------------------------------------------
print('-' * 30)
print('Loading data...')
print('-' * 30)

BASEDIR = '/home/antonio/Downloads/AG_test_RAWIV_proc'

listInputRAWFiles = sorted(glob(BASEDIR + '/*.raw'))


for inputRAWfile in listInputRAWFiles:

    print('\'%s\'...' % (inputRAWfile))

    #image_array = rawpy.imread(inputRAWfile)
    #image_array = np.fromfile(inputRAWfile, dtype=np.int8, sep="")

    image = Image.open(inputRAWfile,'rb')
    imshape = image.size

    print image_array

    image_array = np.reshape(image_array, (418, 512, 512, 3))

    print(image_array)
    print(image_array.shape)
    print(np.amax(image_array))
    print(np.amin(image_array))

    for image_slice in image_array:
        image = Image.frombuffer("I", [512, 512], image_slice.astype('I'), 'raw', 'I', 0, 1)

        print(image_slice)
        print(image_slice.shape)
        print(np.amax(image_slice))
        print(np.amin(image_slice))

        plt.imshow(image)
        plt.show()
    #endfor

#endfor