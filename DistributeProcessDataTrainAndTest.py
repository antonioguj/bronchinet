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



def find_indexes_names_images_files(names_images_type_data, list_images_files):

    indexes_names = []
    for iname in names_images_type_data:
        ifound = False
        for i, ifile in enumerate(list_images_files):
            if iname in ifile:
                indexes_names.append(i)
                ifound = True
                break
        #endfor
        if not ifound:
            message = 'data named: \'%s\' not found' % (iname)
            CatchErrorException(message)

    return indexes_names

def find_element_repeated_two_indexes_names(names_images_type_data_1, names_images_type_data_2):

    list_names_repeated = []
    for ielem in names_images_type_data_1:
        if ielem in names_images_type_data_2:
            list_names_repeated.append(ielem)
    #endfor
    return list_names_repeated



def main(args):

    # ---------- SETTINGS ----------
    nameOrigImagesDataRelPath = 'ProcImagesExperData_FULLLUNG'
    nameOrigMasksDataRelPath  = 'ProcMasksExperData_FULLLUNG'

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



    if (args.distribute_fixed_names):
        print('Split dataset with Fixed Names...')

        names_repeated  = find_element_repeated_two_indexes_names(NAME_IMAGES_TRAINING, NAME_IMAGES_VALIDATION)
        names_repeated += find_element_repeated_two_indexes_names(NAME_IMAGES_TRAINING, NAME_IMAGES_TESTING)
        names_repeated += find_element_repeated_two_indexes_names(NAME_IMAGES_VALIDATION, NAME_IMAGES_TESTING)

        if names_repeated:
            message = "found names repeated in list Training / Validation / Testing names: %s" %(names_repeated)
            CatchErrorException(message)

        indexesTraining   = find_indexes_names_images_files(NAME_IMAGES_TRAINING,   listImagesFiles)
        indexesValidation = find_indexes_names_images_files(NAME_IMAGES_VALIDATION, listImagesFiles)
        indexesTesting    = find_indexes_names_images_files(NAME_IMAGES_TESTING,    listImagesFiles)

        print('Training (%s files)/ Validation (%s files)/ Testing (%s files)...' %(len(indexesTraining),
                                                                                    len(indexesValidation),
                                                                                    len(indexesTesting)))
    else:
        nbTrainingFiles   = int(args.prop_data_training * nbImagesFiles)
        nbValidationFiles = int(args.prop_data_validation * nbImagesFiles)
        nbTestingFiles    = int(args.prop_data_testing * nbImagesFiles)

        print('Training (%s files)/ Validation (%s files)/ Testing (%s files)...' %(nbTrainingFiles,
                                                                                    nbValidationFiles,
                                                                                    nbTestingFiles))
        if (args.distribute_random):
            print('Split dataset Randomly...')
            indexesAllFiles = np.random.choice(range(nbImagesFiles), size=nbImagesFiles, replace=False)
        else:
            print('Split dataset In Order...')
            indexesAllFiles = range(nbImagesFiles)

        indexesTraining  = indexesAllFiles[0:nbTrainingFiles]
        indexesValidation= indexesAllFiles[nbTrainingFiles:nbTrainingFiles+nbValidationFiles]
        indexesTesting   = indexesAllFiles[nbTrainingFiles+nbValidationFiles::]



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
    parser.add_argument('--prop_data_training', type=float, default=PROP_DATA_TRAINING)
    parser.add_argument('--prop_data_validation', type=float, default=PROP_DATA_VALIDATION)
    parser.add_argument('--prop_data_testing', type=float, default=PROP_DATA_TESTING)
    parser.add_argument('--distribute_random', type=str2bool, default=DISTRIBUTE_RANDOM)
    parser.add_argument('--distribute_fixed_names', type=str2bool, default=DISTRIBUTE_FIXED_NAMES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
