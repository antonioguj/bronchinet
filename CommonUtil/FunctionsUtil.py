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
import datetime
import time
import csv
import os
import re
import multiprocessing


def makedir(pathname):
    pathname = pathname.strip()
    pathname = pathname.rstrip("\\")
    isExists = os.path.exists(pathname)
    if not isExists:
        os.makedirs(pathname)
        return True
    else:
        return False

def removedir(pathname):
    os.remove(pathname)

def removefile(filename):
    os.remove(filename)

def makelink(filesrc, linkdest):
    os.symlink(filesrc, linkdest)

def realpathlink(filename):
    return os.path.realpath(filename)

def movedir(pathsrc, pathdest):
    os.rename(pathsrc, pathdest)

def movefile(filesrc, filedest):
    os.rename(filesrc, filedest)

def listfilesDir(pathname):
    listfiles = os.listdir(pathname)
    return [joinpathnames(pathname, file) for file in listfiles]

def isExistdir(pathname):
    return os.path.exists(pathname)

def isExistfile(filename):
    return os.path.exists(filename)

def joinpathnames(pathname, filename):
    return os.path.join(pathname, filename)

def basename(filename):
    return os.path.basename(filename)

def dirnamepathfile(filename):
    return os.path.dirname(filename)

def ospath_splitext_recurse(filename):
    #account for extension that are compound: '.nii.gz'
    basename, extension = os.path.splitext(filename)
    if extension == '':
        return (basename, extension)
    else:
        sub_basename, sub_extension = ospath_splitext_recurse(basename)
        return (sub_basename, sub_extension + extension)

def filenamenoextension(filename):
    return ospath_splitext_recurse(basename(filename))[0]

def filenamepathnoextension(filename):
    return ospath_splitext_recurse(filename)[0]

def filenameextension(filename):
    return ospath_splitext_recurse(filename)[1]

def findFilesDir(filenames):
    return sorted(glob.glob(filenames))

def findFilesDir(filespath, filenames):
    return sorted(glob.glob(joinpathnames(filespath, filenames)))

def saveDictionary(filename, dictionary):
    saveDictionary_numpy(filename, dictionary)

def readDictionary(filename):
    return readDictionary_numpy(filename)

def saveDictionary_numpy(filename, dictionary):
    np.save(filename, dictionary)

def saveDictionary_csv(filename, dictionary):
    if filenameextension(filename) != '.csv':
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
    if filenameextension(filename) != '.csv':
        message = "need \".csv\" to save dictionary"
        CatchErrorException(message)
    else:
        with open(filename, 'r') as fin:
            reader = csv.reader(fin)
            return dict(reader)

def isOddIntegerVal(val):
    return val % 2 == 1
def isEvenIntegerVal(val):
    return val % 2 == 0

def isBiggerTuple(var1, var2):
    return all((v_1 > v_2) for (v_1, v_2) in zip(var1, var2))

def isSmallerTuple(var1, var2):
    return all((v_1 < v_2) for (v_1, v_2) in zip(var1, var2))

def isEqualTuple(var1, var2):
    return all((v_1 == v_2) for (v_1, v_2) in zip(var1, var2))

def sumTwoTuples(var1, var2):
    return tuple(a+b for (a,b) in zip(var1, var2))

def str2bool(strin):
    return strin.lower() in ("yes", "true", "t", "1")

def str2tupleint(strin):
    return tuple([int(i) for i in strin.rsplit(',')])

def str2tuplefloat(strin):
    return tuple([float(i) for i in strin.rsplit(',')])

def list2str(list):
    return "_".join(str(i) for i in list)

def tuple2str(tuple):
    return "_".join(str(i) for i in list(tuple))

def getdatetoday():
    today = datetime.date.today()
    return (today.month, today.day, today.year)

def gettimenow():
    now = datetime.datetime.now()
    return (now.hour, now.minute, now.second)

def splitListInChunks(list, sizechunck):
    listoflists = []
    for i in range(0, len(list), sizechunck):
        listoflists.append( list[i:i+sizechunck] )
    return listoflists

def flattenOutListOfLists(list):
    return reduce(lambda el1, el2: el1 + el2, list)

def mergeTwoListsIntoDictoinary(list1, list2):
    new_dict = {}
    map(lambda key, val: new_dict.update({key: val}), list1, list2)
    return new_dict

def parseListarg(args):
    return args.replace('[','').replace(']','').split(',')

def processBinaryMasks(masks_array):
    # Turn to binary masks (0, 1)
    return np.where(masks_array != 0, 1, 0)

def processBinaryMasks_Type2(masks_array):
    # Turn to binary masks (0, 1)
    return np.where(masks_array == 1, 1, 0)

def processBinaryMasks_KeepExclusion(masks_array):
    # Turn to binary masks (0, 1)
    return np.where(np.logical_or(masks_array != 0, masks_array != -1), 1, 0)

def processMulticlassMasks(masks_array, num_classes):
    # Turn to binary masks (0, 1)
    return np.where(masks_array > num_classes, 0, masks_array)

def checkCorrectNumClassesInMasks(masks_array, num_classes):
    #check that there are as many classes as labels in "masks_array"
    #check also that values are between (0, num_classes)
    return (len(np.unique(masks_array)) == num_classes+1) and (np.amin(masks_array) == 0) and (np.amax(masks_array) == num_classes)

def revertStackImages(images_array):
    # revert to start from traquea
    return images_array[::-1]

def getThresholdImage(image, threshold):
    return np.where(image > threshold, 1, 0)

def getFileExtension(formatoutfile):
    if formatoutfile=='dicom':
        return '.dcm'
    elif formatoutfile=='nifti':
        return '.nii'
    elif formatoutfile=='numpy':
        return '.npy'
    elif formatoutfile=='hdf5':
        return '.hdf5'

def getSavedModelFileName(typeSavedModel):
    if 'model_' not in typeSavedModel:
        typeSavedModel = 'model_' + typeSavedModel
    if '.hdf5' not in typeSavedModel:
        typeSavedModel = typeSavedModel + '.hdf5'
    return typeSavedModel

def getExtractSubstringPattern(string, substr_pattern):
    return re.search(substr_pattern, string).group(0)

def getIndexOriginImagesFile(imagesFile, beginString='images'):
    pattern = beginString + '-[0-9]*'
    if bool(re.match(pattern, imagesFile)):
        return int(re.search(pattern, imagesFile).group(0)[-2:])
    else:
        return False

class WallClockTime(object):
    def __init__(self):
        self.start_time = time.time()
    def compute(self):
        return time.time() - self.start_time

def getNumWorkingProcessesCPU():
    return multiprocessing.cpu_count()


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
