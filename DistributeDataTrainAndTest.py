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

    nameDestImagesFiles   = 'images-%0.2i.dcm'
    nameDestMasksFiles    = 'masks-%0.2i.dcm'
    nameDestAddMasksFiles = 'addMasks-%0.2i.dcm'

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

        nameDestimagesFile   = joinpathnames(TrainingImagesPath,   nameDestImagesFiles%(count_file)  )
        nameDestmasksFile    = joinpathnames(TrainingMasksPath,    nameDestMasksFiles%(count_file)   )
        nameDestaddMasksFile = joinpathnames(TrainingAddMasksPath, nameDestAddMasksFiles%(count_file))

        os.system('ln -s %s %s' % (listImagesFiles[index],  nameDestimagesFile  ))
        os.system('ln -s %s %s' % (listMasksFiles[index],   nameDestmasksFile   ))
        os.system('ln -s %s %s' % (listAddMasksFiles[index],nameDestaddMasksFile))

        count_file += 1
    #endfor
    # ******************** TRAINING DATA ********************


    # ******************** VALIDATION DATA ********************
    for index in indexesValidation:

        nameDestimagesFile   = joinpathnames(ValidationImagesPath,   nameDestImagesFiles%(count_file)  )
        nameDestmasksFile    = joinpathnames(ValidationMasksPath,    nameDestMasksFiles%(count_file)   )
        nameDestaddMasksFile = joinpathnames(ValidationAddMasksPath, nameDestAddMasksFiles%(count_file))

        os.system('ln -s %s %s' % (listImagesFiles[index],  nameDestimagesFile  ))
        os.system('ln -s %s %s' % (listMasksFiles[index],   nameDestmasksFile   ))
        os.system('ln -s %s %s' % (listAddMasksFiles[index],nameDestaddMasksFile))

        count_file += 1
    #endfor
    # ******************** VALIDATION DATA ********************


    # ******************** TESTING DATA ********************
    for i, index in enumerate(indexesTesting):

        nameDestimagesFile   = joinpathnames(TestingImagesPath,   nameDestImagesFiles%(count_file)  )
        nameDestmasksFile    = joinpathnames(TestingMasksPath,    nameDestMasksFiles%(count_file)   )
        nameDestaddMasksFile = joinpathnames(TestingAddMasksPath, nameDestAddMasksFiles%(count_file))

        os.system('ln -s %s %s' % (listImagesFiles[index],  nameDestimagesFile  ))
        os.system('ln -s %s %s' % (listMasksFiles[index],   nameDestmasksFile   ))
        os.system('ln -s %s %s' % (listAddMasksFiles[index],nameDestaddMasksFile))

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