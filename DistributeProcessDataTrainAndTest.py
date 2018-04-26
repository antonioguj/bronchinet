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
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import *
import argparse


def main(args):

    workDirsManager = WorkDirsManager(args.basedir)

    OriginDataPath   = workDirsManager.getNameBaseDataPath()
    OriginImagesPath = workDirsManager.getNameExistPath(OriginDataPath, 'ProcImages')
    OriginMasksPath  = workDirsManager.getNameExistPath(OriginDataPath, 'ProcMasks')

    TrainingDataPath   = workDirsManager.getNameTrainingDataPath()
    TrainingImagesPath = workDirsManager.getNameNewPath(TrainingDataPath, 'ProcImages')
    TrainingMasksPath  = workDirsManager.getNameNewPath(TrainingDataPath, 'ProcMasks')

    ValidationDataPath   = workDirsManager.getNameValidationDataPath()
    ValidationImagesPath = workDirsManager.getNameNewPath(ValidationDataPath, 'ProcImages')
    ValidationMasksPath  = workDirsManager.getNameNewPath(ValidationDataPath, 'ProcMasks')

    TestingDataPath   = workDirsManager.getNameTestingDataPath()
    TestingImagesPath = workDirsManager.getNameNewPath(TestingDataPath, 'ProcImages')
    TestingMasksPath  = workDirsManager.getNameNewPath(TestingDataPath, 'ProcMasks')

    nameOriginImagesFiles = 'images-%0.2i.nii'
    nameOriginMasksFiles  = 'masks-%0.2i.nii'

    listImagesFiles = findFilesDir(OriginImagesPath, nameOriginImagesFiles)
    listMasksFiles  = findFilesDir(OriginMasksPath,  nameOriginMasksFiles)

    nbImagesFiles = len(listImagesFiles)
    nbMasksFiles  = len(listMasksFiles)


    if (nbImagesFiles != nbMasksFiles):
        message = "num Images files not equal to num Masks..."
        CatchErrorException(message)

    nbTrainingFiles   = int(args.prop_training * nbImagesFiles)
    nbValidationFiles = int(args.prop_validation * nbImagesFiles)
    nbTestingFiles    = int(args.prop_testing * nbImagesFiles)

    print('Splitting full dataset in Training, Validation and Testing files...(%s, %s, %s)' %(nbTrainingFiles,
                                                                                              nbValidationFiles,
                                                                                              nbTestingFiles))
    if (args.distribute_random):

        randomIndexes     = np.random.choice(range(nbImagesFiles), size=nbImagesFiles, replace=False)
        indexesTraining   = randomIndexes[0:nbTrainingFiles]
        indexesValidation = randomIndexes[nbTrainingFiles:nbTrainingFiles+nbValidationFiles]
        indexesTesting    = randomIndexes[nbTrainingFiles+nbValidationFiles::]
    else:

        orderedIndexes    = range(nbImagesFiles)
        indexesTraining   = orderedIndexes[0:nbTrainingFiles]
        indexesValidation = orderedIndexes[nbTrainingFiles:nbTrainingFiles+nbValidationFiles]
        indexesTesting    = orderedIndexes[nbTrainingFiles+nbValidationFiles::]

    print('Files assigned to Training Data: %s'   %([basename(listImagesFiles[index]) for index in indexesTraining  ]))
    print('Files assigned to Validation Data: %s' %([basename(listImagesFiles[index]) for index in indexesValidation]))
    print('Files assigned to Testing Data: %s'    %([basename(listImagesFiles[index]) for index in indexesTesting   ]))


    # ******************** TRAINING DATA ********************
    for index in indexesTraining:
        makelink(listImagesFiles[index], joinpathnames(TrainingImagesPath, basename(listImagesFiles[index])))
        makelink(listMasksFiles[index],  joinpathnames(TrainingMasksPath,  basename(listMasksFiles[index])))
    #endfor
    # ******************** TRAINING DATA ********************


    # ******************** VALIDATION DATA ********************
    for index in indexesValidation:
        makelink(listImagesFiles[index], joinpathnames(ValidationImagesPath, basename(listImagesFiles[index])))
        makelink(listMasksFiles[index],  joinpathnames(ValidationMasksPath,  basename(listMasksFiles[index])))
    #endfor
    # ******************** VALIDATION DATA ********************


    # ******************** TESTING DATA ********************
    for index in indexesTesting:
        makelink(listImagesFiles[index], joinpathnames(TestingImagesPath, basename(listImagesFiles[index])))
        makelink(listMasksFiles[index],  joinpathnames(TestingMasksPath,  basename(listMasksFiles[index])))
    #endfor
    # ******************** TESTING DATA ********************


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=DATADIR)
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--prop_training', type=float, default=PROP_TRAINING)
    parser.add_argument('--prop_validation', type=float, default=PROP_VALIDATION)
    parser.add_argument('--prop_testing', type=float, default=PROP_TESTING)
    parser.add_argument('--distribute_random', type=str2bool, default=DISTRIBUTE_RANDOM)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)