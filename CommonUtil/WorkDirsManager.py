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
        self.basePath = basePath + '/'

    mapTypeData_RelDataPath = { 'training' : 'trainingData', 'validation' : 'validationData', 'testing': 'testingData' }

    RelModelsPath    = 'Models'
    RelBestModelPath = 'BestModel'


    def getNameDataPath(self, typedata):
        return self.basePath + self.mapTypeData_RelDataPath[typedata] + '/'

    def getNameTrainingDataPath(self):
        return self.basePath + self.mapTypeData_RelDataPath['training'] + '/'

    def getNameValidationDataPath(self):
        return self.basePath + self.mapTypeData_RelDataPath['validation'] + '/'

    def getNameTestingDataPath(self):
        return self.basePath + self.mapTypeData_RelDataPath['testing'] + '/'

    def getNameRawImagesDataPath(self, typedata):
        return self.getNameDataPath(typedata) + 'Images' + '/'

    def getNameRawGroundTruthDataPath(self, typedata):
        return self.getNameDataPath(typedata) + 'GroundTruth' + '/'

    def getNameModelsPath(self):
        return self.basePath + self.RelModelsPath + '/'

    def getNameBestModelPath(self):
        return self.basePath + self.RelBestModelPath + '/'

    def getNewFullNamePath(self, namepath):
        return self.getNewNamePath(self.basePath + namepath)

    @staticmethod
    def getNewNamePath(namepath):
        if os.path.isdir(namepath):
            count=1
            while( True ):
                if not os.path.isdir(namepath + '_' + count):
                    return namepath + '_' + count
                else:
                    count = count+1
        else:
            return namepath

    @staticmethod
    def getListFilesInPath(path, extension=None):
        if extension:
            sorted( glob(path + '*' + extension))
        else:
            sorted( glob(path) )