#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.FunctionsUtil import *


class WorkDirsManager(object):

    mapTypeData_RelDataPath = {'training'  : 'TrainingData/',
                               'validation': 'ValidationData/',
                               'testing'   : 'TestingData/'}
    RelModelsPath = 'Models/'


    def __init__(self, basePath):
        self.basePath = basePath
        if not isExistdir(basePath):
            message = "WorkDirsManager: base path does not exist..."
            CatchErrorException(message)

    def getNameDataPath(self, typedata):
        return joinpathnames(self.basePath, self.mapTypeData_RelDataPath[typedata])

    def getNameRelDataPath(self, typedata):
        return self.mapTypeData_RelDataPath[typedata]

    def getNameTrainingDataPath(self):
        return joinpathnames(self.basePath, self.mapTypeData_RelDataPath['training'])

    def getNameValidationDataPath(self):
        return joinpathnames(self.basePath, self.mapTypeData_RelDataPath['validation'])

    def getNameTestingDataPath(self):
        return joinpathnames(self.basePath, self.mapTypeData_RelDataPath['testing'])

    def getNameModelsPath(self):
        return joinpathnames(self.basePath, self.RelModelsPath)

    def getNameExistPath(self, basePath, newRelPath):
        newPath = joinpathnames(basePath, newRelPath)
        if not isExistdir(newPath):
            message = "WorkDirsManager: new path \'%s\', does not exist..."%(newPath)
            CatchErrorException(message)
        return newPath

    def getNameNewPath(self, basePath, newRelPath):
        newPath = joinpathnames(basePath, newRelPath)
        if not isExistdir(newPath):
            makedir(newPath)
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