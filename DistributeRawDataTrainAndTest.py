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

    workDirsManager    = WorkDirsManager(args.basedir)
    OriginImagesPath   = workDirsManager.getNameExistPath(args.datadir, 'CTs')
    OriginMasksPath    = workDirsManager.getNameExistPath(args.datadir, 'Airways')
    OriginAddMasksPath = workDirsManager.getNameExistPath(args.datadir, 'Lungs')

    TrainingDataPath     = workDirsManager.getNameTrainingDataPath()
    TrainingImagesPath   = workDirsManager.getNameNewPath(TrainingDataPath, 'RawImages')
    TrainingMasksPath    = workDirsManager.getNameNewPath(TrainingDataPath, 'RawMasks')
    TrainingAddMasksPath = workDirsManager.getNameNewPath(TrainingDataPath, 'RawAddMasks')

    ValidationDataPath     = workDirsManager.getNameValidationDataPath()
    ValidationImagesPath   = workDirsManager.getNameNewPath(ValidationDataPath, 'RawImages')
    ValidationMasksPath    = workDirsManager.getNameNewPath(ValidationDataPath, 'RawMasks')
    ValidationAddMasksPath = workDirsManager.getNameNewPath(ValidationDataPath, 'RawAddMasks')

    TestingDataPath     = workDirsManager.getNameTestingDataPath()
    TestingImagesPath   = workDirsManager.getNameNewPath(TestingDataPath, 'RawImages')
    TestingMasksPath    = workDirsManager.getNameNewPath(TestingDataPath, 'RawMasks')
    TestingAddMasksPath = workDirsManager.getNameNewPath(TestingDataPath, 'RawAddMasks')

    nameOriginImagesFiles   = 'av*.dcm'
    nameOriginMasksFiles    = 'av*surface1.dcm'
    nameOriginAddMasksFiles = 'av*lungs.dcm'

    nameDestinImagesFiles   = 'images-%0.2i.dcm'
    nameDestinMasksFiles    = 'masks-%0.2i.dcm'
    nameDestinAddMasksFiles = 'addMasks-%0.2i.dcm'

    listImagesFiles   = findFilesDir(OriginImagesPath,   nameOriginImagesFiles)
    listMasksFiles    = findFilesDir(OriginMasksPath,    nameOriginMasksFiles)
    listAddMasksFiles = findFilesDir(OriginAddMasksPath, nameOriginAddMasksFiles)

    nbImagesFiles   = len(listImagesFiles)
    nbMasksFiles    = len(listMasksFiles)
    nbAddMasksFiles = len(listAddMasksFiles)


    if (nbImagesFiles != nbMasksFiles or
        nbImagesFiles != nbAddMasksFiles):
        message = "nb Images files not equal..."
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


    count_file = 0

    # ******************** TRAINING DATA ********************
    for index in indexesTraining:
        makelink(listImagesFiles[index],   joinpathnames(TrainingImagesPath,   nameDestinImagesFiles  %(count_file)))
        makelink(listMasksFiles[index],    joinpathnames(TrainingMasksPath,    nameDestinMasksFiles   %(count_file)))
        makelink(listAddMasksFiles[index], joinpathnames(TrainingAddMasksPath, nameDestinAddMasksFiles%(count_file)))
        count_file += 1
    #endfor
    # ******************** TRAINING DATA ********************


    # ******************** VALIDATION DATA ********************
    for index in indexesValidation:
        makelink(listImagesFiles[index],   joinpathnames(ValidationImagesPath,   nameDestinImagesFiles  %(count_file)))
        makelink(listMasksFiles[index],    joinpathnames(ValidationMasksPath,    nameDestinMasksFiles   %(count_file)))
        makelink(listAddMasksFiles[index], joinpathnames(ValidationAddMasksPath, nameDestinAddMasksFiles%(count_file)))
        count_file += 1
    #endfor
    # ******************** VALIDATION DATA ********************


    # ******************** TESTING DATA ********************
    for index in indexesTesting:
        makelink(listImagesFiles[index],   joinpathnames(TestingImagesPath,   nameDestinImagesFiles  %(count_file)))
        makelink(listMasksFiles[index],    joinpathnames(TestingMasksPath,    nameDestinMasksFiles   %(count_file)))
        makelink(listAddMasksFiles[index], joinpathnames(TestingAddMasksPath, nameDestinAddMasksFiles%(count_file)))
        count_file += 1
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