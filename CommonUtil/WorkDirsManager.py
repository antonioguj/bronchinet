#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from glob import glob
import os


class WorkDirsManager(object):

    def __init__(self, basePath):
        self.basePath = basePath

    mapTypeData_RelDataPath = { 'training' : 'TrainingData', 'validation' : 'ValidationData', 'testing': 'TestingData' }

    RelRawImagesPath = 'RawImages'
    RelRawMasksPath  = 'RawMasks'
    RelModelsPath    = 'Models'


    def getNameDataPath(self, typedata):
        return os.path.join(self.basePath, self.mapTypeData_RelDataPath[typedata])

    def getNameTrainingDataPath(self):
        return os.path.join(self.basePath, self.mapTypeData_RelDataPath['training'])

    def getNameValidationDataPath(self):
        return os.path.join(self.basePath, self.mapTypeData_RelDataPath['validation'])

    def getNameTestingDataPath(self):
        return os.path.join(self.basePath, self.mapTypeData_RelDataPath['testing'])

    def getNameRawImagesDataPath(self, typedata):
        return os.path.join(self.getNameDataPath(typedata), self.RelRawImagesPath)

    def getNameRawMasksDataPath(self, typedata):
        return os.path.join(self.getNameDataPath(typedata), self.RelRawMasksPath)

    def getNameModelsPath(self):
        return os.path.join(self.basePath, self.RelModelsPath)

    def getNameNewPath(self, basePath, newRelPath):
        newPath = os.path.join(basePath, newRelPath)
        if( not os.path.exists(newPath) ):
            os.makedirs(newPath)
        return newPath

    #@staticmethod
    #def getNewNamePath(namepath):
    #    if os.path.isdir(namepath):
    #        count=1
    #        while( True ):
    #            if not os.path.isdir(namepath + '_' + count):
    #                return namepath + '_' + count
    #            else:
    #                count = count+1
    #    else:
    #        return namepath

    #@staticmethod
    #def getListFilesInPath(path, extension=None):
    #    if extension:
    #        sorted( glob(path + '*' + extension))
    #    else:
    #        sorted( glob(path) )