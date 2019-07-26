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
    nameInputImagesDataRelPath = 'Images_WorkData/'
    nameInputLabelsDataRelPath = 'Labels_WorkData/'
    nameTrainingDataRelPath    = 'TrainingData/'
    nameValidationDataRelPath  = 'ValidationData/'
    nameTestingDataRelPath     = 'TestingData/'
    nameInputImagesFiles       = 'images*' + getFileExtension(FORMATTRAINDATA)
    nameInputLabelsFiles       = 'labels*' + getFileExtension(FORMATTRAINDATA)
    # ---------- SETTINGS ----------


    workDirsManager     = WorkDirsManager(args.basedir)
    InputImagesDataPath = workDirsManager.getNameExistBaseDataPath(nameInputImagesDataRelPath)
    InputLabelsDataPath = workDirsManager.getNameExistBaseDataPath(nameInputLabelsDataRelPath)
    TrainingDataPath    = workDirsManager.getNameNewPath          (nameTrainingDataRelPath)
    ValidationDataPath  = workDirsManager.getNameNewPath          (nameValidationDataRelPath)
    TestingDataPath     = workDirsManager.getNameNewPath          (nameTestingDataRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesDataPath, nameInputImagesFiles)
    listInputLabelsFiles = findFilesDirAndCheck(InputLabelsDataPath, nameInputLabelsFiles)

    if (len(listInputImagesFiles) != len(listInputLabelsFiles)):
        message = 'num images in dir \'%s\', not equal to num labels in dir \'%i\'...' %(len(listInputImagesFiles),
                                                                                         len(listInputLabelsFiles))
        CatchErrorException(message)


    if (args.distribute_fixed_names):
        print("Split dataset with Fixed Names...")
        names_repeated  = find_element_repeated_two_indexes_names(NAME_IMAGES_TRAINING,   NAME_IMAGES_VALIDATION)
        names_repeated += find_element_repeated_two_indexes_names(NAME_IMAGES_TRAINING,   NAME_IMAGES_TESTING)
        names_repeated += find_element_repeated_two_indexes_names(NAME_IMAGES_VALIDATION, NAME_IMAGES_TESTING)

        if names_repeated:
            message = "found names repeated in list Training / Validation / Testing names: %s" %(names_repeated)
            CatchErrorException(message)

        indexes_training_files   = find_indexes_names_images_files(NAME_IMAGES_TRAINING,   listInputImagesFiles)
        indexes_validation_files = find_indexes_names_images_files(NAME_IMAGES_VALIDATION, listInputImagesFiles)
        indexes_testing_files    = find_indexes_names_images_files(NAME_IMAGES_TESTING,    listInputImagesFiles)
        print("Training (%s files)/ Validation (%s files)/ Testing (%s files)..." %(len(indexes_training_files),
                                                                                    len(indexes_validation_files),
                                                                                    len(indexes_testing_files)))
    else:
        numTrainingFiles   = int(args.prop_data_training   * numImagesFiles)
        numValidationFiles = int(args.prop_data_validation * numImagesFiles)
        numTestingFiles    = int(args.prop_data_testing    * numImagesFiles)
        print("Training (%s files)/ Validation (%s files)/ Testing (%s files)..." %(numTrainingFiles,
                                                                                    numValidationFiles,
                                                                                    numTestingFiles))
        if (args.distribute_random):
            print("Split dataset Randomly...")
            indexesAllFiles = np.random.choice(range(numImagesFiles), size=numImagesFiles, replace=False)
        else:
            print("Split dataset In Order...")
            indexesAllFiles = range(numImagesFiles)

        indexes_training_files   = indexesAllFiles[0:numTrainingFiles]
        indexes_validation_files = indexesAllFiles[numTrainingFiles:numTrainingFiles+numValidationFiles]
        indexes_testing_files    = indexesAllFiles[numTrainingFiles+numValidationFiles::]


    # TRAINING DATA
    print("Files assigned to Training Data:")
    for index in indexes_training_files:
        print("%s" %(listInputImagesFiles[index]))
        makelink(listInputImagesFiles[index], joinpathnames(TrainingDataPath, basename(listInputImagesFiles[index])))
        makelink(listInputLabelsFiles[index], joinpathnames(TrainingDataPath, basename(listInputLabelsFiles[index])))
    #endfor

    # VALIDATION DATA
    print("Files assigned to Validation Data:")
    for index in indexes_validation_files:
        print("%s" % (listInputImagesFiles[index]))
        makelink(listInputImagesFiles[index], joinpathnames(ValidationDataPath, basename(listInputImagesFiles[index])))
        makelink(listInputLabelsFiles[index], joinpathnames(ValidationDataPath, basename(listInputLabelsFiles[index])))
    #endfor

    # TESTING DATA
    print("Files assigned to Testing Data:")
    for index in indexes_testing_files:
        print("%s" % (listInputImagesFiles[index]))
        makelink(listInputImagesFiles[index], joinpathnames(TestingDataPath, basename(listInputImagesFiles[index])))
        makelink(listInputLabelsFiles[index], joinpathnames(TestingDataPath, basename(listInputLabelsFiles[index])))
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
