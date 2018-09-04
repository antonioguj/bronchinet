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

    # ---------- SETTINGS ----------
    nameOrigImagesDataRelPath = 'ProcImagesData'
    nameOrigMasksDataRelPath  = 'ProcMasksData'

    nameOriginImagesFiles = 'images*'+ getFileExtension(FORMATINOUTDATA)
    nameOriginMasksFiles  = 'masks*' + getFileExtension(FORMATINOUTDATA)
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)

    OrigImagesDataPath = workDirsManager.getNameExistPath(workDirsManager.getNameBaseDataPath(), nameOrigImagesDataRelPath)
    OrigMasksDataPath  = workDirsManager.getNameExistPath(workDirsManager.getNameBaseDataPath(), nameOrigMasksDataRelPath )
    TrainingDataPath   = workDirsManager.getNameNewPath(workDirsManager.getNameTrainingDataPath())
    ValidationDataPath = workDirsManager.getNameNewPath(workDirsManager.getNameValidationDataPath())
    TestingDataPath    = workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath())

    listImagesFiles = findFilesDir(OrigImagesDataPath, nameOriginImagesFiles)
    listMasksFiles  = findFilesDir(OrigMasksDataPath,  nameOriginMasksFiles)

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
        makelink(listImagesFiles[index], joinpathnames(TrainingDataPath, basename(listImagesFiles[index])))
        makelink(listMasksFiles[index],  joinpathnames(TrainingDataPath,  basename(listMasksFiles[index])))
    #endfor
    # ******************** TRAINING DATA ********************


    # ******************** VALIDATION DATA ********************
    for index in indexesValidation:
        makelink(listImagesFiles[index], joinpathnames(ValidationDataPath, basename(listImagesFiles[index])))
        makelink(listMasksFiles[index],  joinpathnames(ValidationDataPath,  basename(listMasksFiles[index])))
    #endfor
    # ******************** VALIDATION DATA ********************


    # ******************** TESTING DATA ********************
    for index in indexesTesting:
        makelink(listImagesFiles[index], joinpathnames(TestingDataPath, basename(listImagesFiles[index])))
        makelink(listMasksFiles[index],  joinpathnames(TestingDataPath,  basename(listMasksFiles[index])))
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
