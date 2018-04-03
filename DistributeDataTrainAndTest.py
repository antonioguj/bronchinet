#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from CommonUtil.FileReaders import *
from CommonUtil.ErrorMessages import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import WorkDirsManager
import numpy as np



#MAIN
workDirsManager   = WorkDirsManager(BASEDIR)
OriginCTsPath     = workDirsManager.getNameNewPath(DATADIR, 'CTs')
OriginAirwaysPath = workDirsManager.getNameNewPath(DATADIR, 'Airways')
OriginLungsPath   = workDirsManager.getNameNewPath(DATADIR, 'Lungs')

TrainingDataPath   = workDirsManager.getNameTrainingDataPath()
ValidationDataPath = workDirsManager.getNameValidationDataPath()
TestingDataPath    = workDirsManager.getNameTestingDataPath()

TrainingCTsPath     = workDirsManager.getNameNewPath(TrainingDataPath, 'RawCTs')
TrainingAirwaysPath = workDirsManager.getNameNewPath(TrainingDataPath, 'RawAirways')
TrainingLungsPath   = workDirsManager.getNameNewPath(TrainingDataPath, 'RawLungs')
TrainingImagesPath  = workDirsManager.getNameNewPath(TrainingDataPath, 'RawImages')
TrainingMasksPath   = workDirsManager.getNameNewPath(TrainingDataPath, 'RawMasks')

ValidationCTsPath     = workDirsManager.getNameNewPath(ValidationDataPath, 'RawCTs')
ValidationAirwaysPath = workDirsManager.getNameNewPath(ValidationDataPath, 'RawAirways')
ValidationLungsPath   = workDirsManager.getNameNewPath(ValidationDataPath, 'RawLungs')
ValidationImagesPath  = workDirsManager.getNameNewPath(ValidationDataPath, 'RawImages')
ValidationMasksPath   = workDirsManager.getNameNewPath(ValidationDataPath, 'RawMasks')

TestingCTsPath     = workDirsManager.getNameNewPath(TestingDataPath, 'RawCTs')
TestingAirwaysPath = workDirsManager.getNameNewPath(TestingDataPath, 'RawAirways')
TestingLungsPath   = workDirsManager.getNameNewPath(TestingDataPath, 'RawLungs')
TestingImagesPath  = workDirsManager.getNameNewPath(TestingDataPath, 'RawImages')
TestingMasksPath   = workDirsManager.getNameNewPath(TestingDataPath, 'RawMasks')

nameCTsFiles             = 'av*.dcm'
nameAirwaysLumenFiles    = 'av*surface0.dcm'
nameAirwaysOuterWallFiles= 'av*surface1.dcm'
nameLungsFiles           = 'av*lungs.dcm'


listCTsFiles             = findFilesDir(OriginCTsPath     + nameCTsFiles)
listAirwaysLumenFiles    = findFilesDir(OriginAirwaysPath + nameAirwaysLumenFiles)
listAirwaysOuterWallFiles= findFilesDir(OriginAirwaysPath + nameAirwaysOuterWallFiles)
listLungsFiles           = findFilesDir(OriginLungsPath   + nameLungsFiles)

nbCTsFiles              = len(listCTsFiles)
nbAirwaysLumenFiles     = len(listAirwaysLumenFiles)
nbAirwaysOuterWallFiles = len(listAirwaysOuterWallFiles)
nbLungsFiles            = len(listLungsFiles)


if (nbCTsFiles != nbAirwaysLumenFiles or
    nbCTsFiles != nbAirwaysOuterWallFiles or
    nbCTsFiles != nbLungsFiles):
    message = "nb Images files not equal..."
    CatchErrorException(message)


nbTrainingFiles   = int(PROP_TRAINING  * nbCTsFiles)
nbValidationFiles = int(PROP_VALIDATION* nbCTsFiles)
nbTestingFiles    = int(PROP_TESTING   * nbCTsFiles)

print('Splitting full dataset in Training, Validation and Testing files...(%s, %s, %s)' %(nbTrainingFiles,
                                                                                          nbValidationFiles,
                                                                                          nbTestingFiles))

if (DISTRIBUTE_RANDOM):

    randomIndexes     = np.random.choice(range(nbCTsFiles), size=nbCTsFiles, replace=False)
    indexesTraining   = randomIndexes[0:nbTrainingFiles]
    indexesValidation = randomIndexes[nbTrainingFiles:nbTrainingFiles+nbValidationFiles]
    indexesTesting    = randomIndexes[nbTrainingFiles+nbValidationFiles::]
else:

    orderedIndexes    = range(nbCTsFiles)
    indexesTraining   = orderedIndexes[0:nbTrainingFiles]
    indexesValidation = orderedIndexes[nbTrainingFiles:nbTrainingFiles+nbValidationFiles]
    indexesTesting    = orderedIndexes[nbTrainingFiles+nbValidationFiles::]

print('Files assigned to Training Data: %s'   %([basename(listCTsFiles[index]) for index in indexesTraining  ]))
print('Files assigned to Validation Data: %s' %([basename(listCTsFiles[index]) for index in indexesValidation]))
print('Files assigned to Testing Data: %s'    %([basename(listCTsFiles[index]) for index in indexesTesting   ]))



# ******************** TRAINING DATA ********************
for index in indexesTraining:

    name_dest_CT_file               = joinpathnames(TrainingCTsPath,     basename(listCTsFiles[index])             )
    name_dest_AirwaysLumen_file     = joinpathnames(TrainingAirwaysPath, basename(listAirwaysLumenFiles[index])    )
    name_dest_AirwaysOuterWall_file = joinpathnames(TrainingAirwaysPath, basename(listAirwaysOuterWallFiles[index]))
    name_dest_Lungs_file            = joinpathnames(TrainingLungsPath,   basename(listLungsFiles[index])           )

    os.system('ln -s %s %s' % (listCTsFiles[index],             name_dest_CT_file              ))
    os.system('ln -s %s %s' % (listAirwaysLumenFiles[index],    name_dest_AirwaysLumen_file    ))
    os.system('ln -s %s %s' % (listAirwaysOuterWallFiles[index],name_dest_AirwaysOuterWall_file))
    os.system('ln -s %s %s' % (listLungsFiles[index],           name_dest_Lungs_file           ))
#endfor

listImagesFiles = findFilesDir(TrainingCTsPath     + nameCTsFiles)
listMasksFiles  = findFilesDir(TrainingAirwaysPath + nameAirwaysOuterWallFiles)

for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

    name_dest_Images_file = joinpathnames(TrainingImagesPath, basename(imagesFile))
    name_dest_Masks_file  = joinpathnames(TrainingMasksPath,  basename(masksFile ))

    os.system('ln -s %s %s' % (imagesFile, name_dest_Images_file))
    os.system('ln -s %s %s' % (masksFile,  name_dest_Masks_file ))
#endfor
# ******************** TRAINING DATA ********************



# ******************** VALIDATION DATA ********************
for index in indexesValidation:

    name_dest_CT_file               = joinpathnames(ValidationCTsPath,     basename(listCTsFiles[index])             )
    name_dest_AirwaysLumen_file     = joinpathnames(ValidationAirwaysPath, basename(listAirwaysLumenFiles[index])    )
    name_dest_AirwaysOuterWall_file = joinpathnames(ValidationAirwaysPath, basename(listAirwaysOuterWallFiles[index]))
    name_dest_Lungs_file            = joinpathnames(ValidationLungsPath,   basename(listLungsFiles[index])           )

    os.system('ln -s %s %s' % (listCTsFiles[index],             name_dest_CT_file              ))
    os.system('ln -s %s %s' % (listAirwaysLumenFiles[index],    name_dest_AirwaysLumen_file    ))
    os.system('ln -s %s %s' % (listAirwaysOuterWallFiles[index],name_dest_AirwaysOuterWall_file))
    os.system('ln -s %s %s' % (listLungsFiles[index],           name_dest_Lungs_file           ))
#endfor

listImagesFiles = findFilesDir(ValidationCTsPath     + nameCTsFiles)
listMasksFiles  = findFilesDir(ValidationAirwaysPath + nameAirwaysOuterWallFiles)

for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

    name_dest_Images_file = joinpathnames(ValidationImagesPath, basename(imagesFile))
    name_dest_Masks_file  = joinpathnames(ValidationMasksPath,  basename(masksFile ))

    os.system('ln -s %s %s' % (imagesFile, name_dest_Images_file))
    os.system('ln -s %s %s' % (masksFile,  name_dest_Masks_file ))
#endfor
# ******************** VALIDATION DATA ********************



# ******************** TESTING DATA ********************
for index in indexesTesting:

    name_dest_CT_file               = joinpathnames(TestingCTsPath,     basename(listCTsFiles[index])             )
    name_dest_AirwaysLumen_file     = joinpathnames(TestingAirwaysPath, basename(listAirwaysLumenFiles[index])    )
    name_dest_AirwaysOuterWall_file = joinpathnames(TestingAirwaysPath, basename(listAirwaysOuterWallFiles[index]))
    name_dest_Lungs_file            = joinpathnames(TestingLungsPath,   basename(listLungsFiles[index])           )

    os.system('ln -s %s %s' % (listCTsFiles[index],             name_dest_CT_file              ))
    os.system('ln -s %s %s' % (listAirwaysLumenFiles[index],    name_dest_AirwaysLumen_file    ))
    os.system('ln -s %s %s' % (listAirwaysOuterWallFiles[index],name_dest_AirwaysOuterWall_file))
    os.system('ln -s %s %s' % (listLungsFiles[index],           name_dest_Lungs_file           ))
#endfor

listImagesFiles = findFilesDir(TestingCTsPath     + nameCTsFiles)
listMasksFiles  = findFilesDir(TestingAirwaysPath + nameAirwaysOuterWallFiles)

for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

    name_dest_Images_file = joinpathnames(TestingImagesPath, basename(imagesFile))
    name_dest_Masks_file  = joinpathnames(TestingMasksPath,  basename(masksFile ))

    os.system('ln -s %s %s' % (imagesFile, name_dest_Images_file))
    os.system('ln -s %s %s' % (masksFile,  name_dest_Masks_file ))
# endfor
# ******************** TESTING DATA ********************