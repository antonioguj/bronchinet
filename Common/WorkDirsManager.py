#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.FunctionsUtil import *


class WorkDirsManager(object):

    baseDataRelPath = 'BaseData/'
    mapTypeData_RelDataPath = {'training'  : 'TrainingData/',
                               'validation': 'ValidationData/',
                               'testing'   : 'TestingData/'}
    modelsRelPath = 'Models/'


    def __init__(self, basePath):
        #self.basePath = basePath
        # add cwd to get full path
        self.basePath = joinpathnames(currentdir(), basePath)
        if not isExistdir(self.basePath):
            message = "WorkDirsManager: base path \'%s\' does not exist..." %(self.basePath)
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

    def getNameBaseDataPath(self):
        return joinpathnames(self.basePath, self.baseDataRelPath)

    def getNameModelsPath(self):
        return joinpathnames(self.basePath, self.modelsRelPath)


    def getNameExistPath(self, relPath):
        fullPath = joinpathnames(self.basePath, relPath)
        if not isExistdir(fullPath):
            message = "WorkDirsManager: path \'%s\', does not exist..."%(fullPath)
            CatchErrorException(message)
        return fullPath

    def getNameNewPath(self, relPath):
        fullPath = joinpathnames(self.basePath, relPath)
        if not isExistdir(fullPath):
            makedir(fullPath)
        return fullPath

    def getNameNewUpdatePath(self, relPath):
        newfullPath = makeUpdatedir(joinpathnames(self.basePath, relPath))
        return newfullPath

    def getNameExistBaseDataPath(self, relPath):
        return self.getNameExistPath(joinpathnames(self.baseDataRelPath, relPath))

    def getNameNewBaseDataPath(self, relPath):
        return self.getNameNewPath(joinpathnames(self.baseDataRelPath, relPath))


    def getNameExistFile(self, filename):
        fullFilename = joinpathnames(self.basePath, filename)
        if not isExistfile(fullFilename):
            message = "WorkDirsManager: file \'%s\', does not exist..."%(fullFilename)
            CatchErrorException(message)
        return fullFilename

    def getNameNewFile(self, filename):
        fullFilename = joinpathnames(self.basePath, filename)
        return fullFilename

    def getNameNewUpdateFile(self, filename):
        newfullFilename = newUpdatefile(joinpathnames(self.basePath, filename))
        return newfullFilename

    def getNameExistBaseDataFile(self, filename):
        return self.getNameExistFile(joinpathnames(self.baseDataRelPath, filename))

    def getNameNewBaseDataFile(self, filename):
        return self.getNameNewFile(joinpathnames(self.baseDataRelPath, filename))