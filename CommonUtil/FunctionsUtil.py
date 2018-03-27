#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.ErrorMessages import *
import numpy as np
import glob
import csv
import os


def mkdir(pathname):
    pathname = pathname.strip()
    pathname = pathname.rstrip("\\")
    isExists = os.path.exists(pathname)
    if not isExists:
        os.makedirs(pathname)
        return True
    else:
        return False

def isExistfile(filename):
    return os.path.exists(filename)

def joinpathnames(pathname, filename):
    return os.path.join(pathname, filename)

def basename(filename):
    return os.path.basename(filename)

def filenamenoextension(filename):
    return os.path.splitext(filename)[0]

def fileextension(filename):
    return os.path.splitext(filename)[1]


def splitListInChunks(list, sizechunck):

    listoflists = []
    for i in range(0, len(list), sizechunck):
        listoflists.append( list[i:i+sizechunck] )
    return listoflists

def findFilesDir(filenames):
    return sorted(glob.glob(filenames))


def saveDictionary(filename, dictionary):
    saveDictionary_numpy(filename, dictionary)

def readDictionary(filename):
    return readDictionary_numpy(filename)


def saveDictionary_numpy(filename, dictionary):
    np.save(filename, dictionary)

def saveDictionary_csv(filename, dictionary):
    if fileextension(filename) != '.csv':
        message = "need \".csv\" to save dictionary"
        CatchErrorException(message)
    else:
        with open(filename, 'w') as fout:
            writer = csv.writer(fout)
            for key, value in dictionary.items():
                writer.writerow([key, value])

def readDictionary_numpy(filename):
    return np.load(filename).item()

def readDictionary_csv(filename):
    if fileextension(filename) != '.csv':
        message = "need \".csv\" to save dictionary"
        CatchErrorException(message)
    else:
        with open(filename, 'r') as fin:
            reader = csv.reader(fin)
            return dict(reader)


def processBinaryMasks(masks_array):
    # Turn to binary masks (0, 1)
    return np.where(masks_array != 0, 1, 0)


def processBinaryMasks_KeepExclusion(masks_array):
    # Turn to binary masks (0, 1)
    return np.where(masks_array != 0 or masks_array != -1, 1, 0)


def revertStackImages(images_array):
    # revert to start from traquea
    return images_array[::-1]


def getThresholdImage(image, threshold):
    return np.where(image > threshold, 1, 0)


# Manage GPU resources
DEFAULT_FRAC_GPU_USAGE = 0.5

def set_limit_gpu_memory_usage(frac_gpu_usage=DEFAULT_FRAC_GPU_USAGE):

    from keras import backend as k
    import tensorflow as tf

    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = frac_gpu_usage
    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))