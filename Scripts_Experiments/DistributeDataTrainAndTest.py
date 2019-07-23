#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Common.FunctionsUtil import *
from Common.WorkDirsManager import *
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
            message = 'data named: \'%s\' not found' %(iname)
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
    nameOriginImagesDataRelPath = 'Images_WorkData/'
    nameOriginLabelsDataRelPath = 'Labels_WorkData/'
    nameOriginImagesFiles       = 'images*' + getFileExtension(FORMATTRAINDATA)
    nameOriginLabelsFiles       = 'labels*' + getFileExtension(FORMATTRAINDATA)
    # ---------- SETTINGS ----------


    workDirsManager      = WorkDirsManager(args.basedir)
    OriginImagesDataPath = workDirsManager.getNameExistBasePath(nameOriginImagesDataRelPath)
    OriginLabelsDataPath = workDirsManager.getNameExistBasePath(nameOriginLabelsDataRelPath)
    TrainingDataPath     = workDirsManager.getNameNewPath      ('TrainingData/')
    ValidationDataPath   = workDirsManager.getNameNewPath      ('ValidationData/')
    TestingDataPath      = workDirsManager.getNameNewPath      ('TestingData/')

    listImagesFiles = findFilesDir(OriginImagesDataPath, nameOriginImagesFiles)
    listLabelsFiles = findFilesDir(OriginLabelsDataPath, nameOriginLabelsFiles)

    numImagesFiles = len(listImagesFiles)
    numLabelsFiles = len(listLabelsFiles)

    if (numImagesFiles != numLabelsFiles):
        message = "num image files \'%s\' not equal to num ground-truth files \'%s\'..." %(numImagesFiles, numLabelsFiles)
        CatchErrorException(message)


    if (args.distribute_fixed_names):
        print("Split dataset with Fixed Names...")
        names_repeated  = find_element_repeated_two_indexes_names(NAME_IMAGES_TRAINING, NAME_IMAGES_VALIDATION)
        names_repeated += find_element_repeated_two_indexes_names(NAME_IMAGES_TRAINING, NAME_IMAGES_TESTING)
        names_repeated += find_element_repeated_two_indexes_names(NAME_IMAGES_VALIDATION, NAME_IMAGES_TESTING)

        if names_repeated:
            message = "found names repeated in list Training / Validation / Testing names: %s" %(names_repeated)
            CatchErrorException(message)

        indexesTraining = find_indexes_names_images_files(NAME_IMAGES_TRAINING, listImagesFiles)
        indexesValidation = find_indexes_names_images_files(NAME_IMAGES_VALIDATION, listImagesFiles)
        indexesTesting = find_indexes_names_images_files(NAME_IMAGES_TESTING, listImagesFiles)
        print("Training (%s files)/ Validation (%s files)/ Testing (%s files)..." %(len(indexesTraining),
                                                                                    len(indexesValidation),
                                                                                    len(indexesTesting)))
    else:
        numTrainingFiles = int(args.prop_data_training * numImagesFiles)
        numValidationFiles = int(args.prop_data_validation * numImagesFiles)
        numTestingFiles = int(args.prop_data_testing * numImagesFiles)
        print("Training (%s files)/ Validation (%s files)/ Testing (%s files)..." %(numTrainingFiles,
                                                                                    numValidationFiles,
                                                                                    numTestingFiles))
        if (args.distribute_random):
            print("Split dataset Randomly...")
            indexesAllFiles = np.random.choice(range(numImagesFiles), size=numImagesFiles, replace=False)
        else:
            print("Split dataset In Order...")
            indexesAllFiles = range(numImagesFiles)

        indexesTraining = indexesAllFiles[0:numTrainingFiles]
        indexesValidation = indexesAllFiles[numTrainingFiles:numTrainingFiles+numValidationFiles]
        indexesTesting = indexesAllFiles[numTrainingFiles+numValidationFiles::]


    print("Files assigned to Training Data: \'%s\'" %([basename(listImagesFiles[index]) for index in indexesTraining]))
    print("Files assigned to Validation Data: \'%s\'" %([basename(listImagesFiles[index]) for index in indexesValidation]))
    print("Files assigned to Testing Data: \'%s\'" %([basename(listImagesFiles[index]) for index in indexesTesting]))

    # ******************** TRAINING DATA ********************
    for index in indexesTraining:
        makelink(listImagesFiles[index], joinpathnames(TrainingDataPath, basename(listImagesFiles[index])))
        makelink(listLabelsFiles[index], joinpathnames(TrainingDataPath, basename(listLabelsFiles[index])))
    #endfor
    # ******************** TRAINING DATA ********************

    # ******************** VALIDATION DATA ********************
    for index in indexesValidation:
        makelink(listImagesFiles[index], joinpathnames(ValidationDataPath, basename(listImagesFiles[index])))
        makelink(listLabelsFiles[index], joinpathnames(ValidationDataPath, basename(listLabelsFiles[index])))
    #endfor
    # ******************** VALIDATION DATA ********************

    # ******************** TESTING DATA ********************
    for index in indexesTesting:
        makelink(listImagesFiles[index], joinpathnames(TestingDataPath, basename(listImagesFiles[index])))
        makelink(listLabelsFiles[index], joinpathnames(TestingDataPath, basename(listLabelsFiles[index])))
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
