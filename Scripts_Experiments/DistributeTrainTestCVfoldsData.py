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



def main(args):
    # ---------- SETTINGS ----------
    nameInputImagesDataRelPath = 'ImagesWorkData/'
    nameInputLabelsDataRelPath = 'LabelsWorkData/'
    nameReferenceFilesRelPath  = 'Images/'
    nameTrainingAllDataRelPath = 'TrainingAllData/'
    nameTestingAllDataRelPath  = 'TestingAllData/'
    nameTrainDataSubRelPath    = 'Train-CV%0.2i/'
    nameTestDataSubRelPath     = 'Test-CV%0.2i/'
    nameCVfoldsRelPath         = 'CV-folds/'
    nameCVfoldsFiles           = 'test[0-9].txt'
    # ---------- SETTINGS ----------


    workDirsManager     = WorkDirsManager(args.basedir)
    InputImagesDataPath = workDirsManager.getNameExistBaseDataPath(nameInputImagesDataRelPath)
    InputLabelsDataPath = workDirsManager.getNameExistBaseDataPath(nameInputLabelsDataRelPath)
    ReferenceFilesPath  = workDirsManager.getNameExistBaseDataPath(nameReferenceFilesRelPath)
    CVfoldsPath         = workDirsManager.getNameExistPath        (nameCVfoldsRelPath)
    TrainingAllDataPath = workDirsManager.getNameNewPath          (nameTrainingAllDataRelPath)
    TestingAllDataPath  = workDirsManager.getNameNewPath          (nameTestingAllDataRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesDataPath)
    listInputLabelsFiles = findFilesDirAndCheck(InputLabelsDataPath)
    listReferenceFiles   = findFilesDirAndCheck(ReferenceFilesPath)
    listReferenceFiles   = [basename(elem) for elem in listReferenceFiles]  # create list with only basenames
    listCVfoldsFiles     = findFilesDirAndCheck(CVfoldsPath, nameCVfoldsFiles)

    if (len(listInputImagesFiles) != len(listInputLabelsFiles)):
        message = 'num images in dir \'%s\', not equal to num labels in dir \'%i\'...' %(len(listInputImagesFiles),
                                                                                         len(listInputLabelsFiles))
        CatchErrorException(message)

    num_imagedata_files = len(listInputImagesFiles)


    for i, in_cvfold_file in enumerate(listCVfoldsFiles):
        print("\nInput: \'%s\'..." % (basename(in_cvfold_file)))
        print("Create Training and Testing sets for the CV fold %i..." %(i))

        TrainingDataPath = joinpathnames(TrainingAllDataPath, nameTrainDataSubRelPath%(i+1))
        TestingDataPath  = joinpathnames(TestingAllDataPath,  nameTestDataSubRelPath %(i+1))

        makedir(TrainingDataPath)
        makedir(TestingDataPath)


        fout = open(in_cvfold_file, 'r')
        in_cvfold_testfile_names = [elem.replace('\n','.nii.gz') for elem in fout.readlines()]

        indexes_testing_files = []
        for icvfold_testfile in in_cvfold_testfile_names:
            index_tesfile = listReferenceFiles.index(icvfold_testfile)
            indexes_testing_files.append(index_tesfile)
        #endfor
        indexes_training_files = [ind for ind in range(num_imagedata_files) if ind not in indexes_testing_files]


        # TRAINING DATA
        print("Files assigned to Training Data:")
        for index in indexes_training_files:
            basename_input_images_file = basename(listInputImagesFiles[index])
            basename_input_labels_file = basename(listInputLabelsFiles[index])
            print("%s --> %s" %(basename_input_images_file, listInputImagesFiles[index]))

            makelink(listInputImagesFiles[index], joinpathnames(TrainingDataPath, basename_input_images_file))
            makelink(listInputLabelsFiles[index], joinpathnames(TrainingDataPath, basename_input_labels_file))
        # endfor

        # TESTING DATA
        print("Files assigned to Testing Data:")
        for index in indexes_testing_files:
            basename_input_images_file = basename(listInputImagesFiles[index])
            basename_input_labels_file = basename(listInputLabelsFiles[index])
            print("%s --> %s" %(basename_input_images_file, listInputImagesFiles[index]))

            makelink(listInputImagesFiles[index], joinpathnames(TestingDataPath, basename_input_images_file))
            makelink(listInputLabelsFiles[index], joinpathnames(TestingDataPath, basename_input_labels_file))
        # endfor
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)