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
        self.basePath = basePath
        if not isExistdir(basePath):
            message = "WorkDirsManager: base path \'%s\' does not exist..." %(basePath)
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
        newPath = joinpathnames(self.basePath, relPath)
        if not isExistdir(newPath):
            message = "WorkDirsManager: path \'%s\', does not exist..."%(newPath)
            CatchErrorException(message)
        return newPath

    def getNameExistBaseDataPath(self, relPath):
        relPath = joinpathnames(self.baseDataRelPath, relPath)
        return self.getNameExistPath(relPath)

    def getNameNewPath(self, relPath):
        newPath = joinpathnames(self.basePath, relPath)
        if not isExistdir(newPath):
            makedir(newPath)
        return newPath

    def getNameNewBaseDataPath(self, relPath):
        relPath = joinpathnames(self.baseDataRelPath, relPath)
        return self.getNameNewPath(relPath)

    def getNameUpdatePath(self, relPath):
        #datetoday_str= '%i-%i-%i'%(getdatetoday())
        #timenow_str  = '%0.2i-%0.2i-%0.2i'%(gettimenow())
        suffix_update = '_NEW%0.2i'
        updatePath = joinpathnames(self.basePath, relPath)
        if isExistdir(updatePath):
            count = 1
            while True:
                newUpdatePath = updatePath + suffix_update%(count)
                if not isExistdir(newUpdatePath):
                    makedir(newUpdatePath)
                    return newUpdatePath
                #else:
                #...keep iterating
                count = count + 1
        else:
            makedir(updatePath)
            return updatePath


    @staticmethod
    def getNameExistFullPath(fullPath):
        if not isExistdir(fullPath):
            message = "WorkDirsManager: path \'%s\', does not exist..."%(fullPath)
            CatchErrorException(message)
        return fullPath

    @staticmethod
    def getNameNewFullPath(fullPath):
        if not isExistdir(fullPath):
            makedir(fullPath)
        return fullPath


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