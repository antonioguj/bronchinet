#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from common.constant import *
from common.functionutil import *
from common.workdirmanager import *
import math
import argparse


def searchIndexesInputFilesFromReferKeysInFile(in_readfile, list_input_referKeys):
    if not is_exist_file(in_readfile):
        message = 'File to specify (Train / Valid / Test) data \'infileorder\ not found: \'%s\'...' % (in_readfile)
        catch_warning_exception(message)
        return []

    out_indexes_input_files = []
    with open(in_readfile, 'r') as fin:
        for in_referkey_file in fin.readlines():
            in_referkey_file = in_referkey_file.replace('\n','').replace('\r','')

            if in_referkey_file in list_input_referKeys:
                index_pos_referkey_file = [ind for (ind, it_file) in enumerate(list_input_referKeys) if it_file==in_referkey_file]
                out_indexes_input_files += index_pos_referkey_file
            else:
                message = '\'%s\' not found in list of Input Reference Keys: \'%s\'...' %(in_referkey_file, list_input_referKeys)
                catch_error_exception(message)
    # --------------------------------------
    return out_indexes_input_files


LIST_TYPEDATA_AVAIL  = ['training', 'testing']
LIST_TYPESDISTDATA_AVAIL = ['original', 'random', 'orderfile']



def main(args):

    workDirsManager     = GeneralDirManager(args.basedir)
    InputImagesDataPath = workDirsManager.getNameExistBaseDataPath(args.nameInputImagesRelPath)
    InputReferKeysFile  = workDirsManager.getNameExistBaseDataFile(args.nameInputReferKeysFile)
    TrainingDataPath    = workDirsManager.get_pathdir_new          (args.nameTrainingDataRelPath)
    ValidationDataPath  = workDirsManager.get_pathdir_new          (args.nameValidationDataRelPath)
    TestingDataPath     = workDirsManager.get_pathdir_new          (args.nameTestingDataRelPath)

    listInputImagesFiles = list_files_dir(InputImagesDataPath)

    if (args.isPrepareLabels):
        InputLabelsDataPath  = workDirsManager.getNameExistBaseDataPath(args.nameInputLabelsRelPath)
        listInputLabelsFiles = list_files_dir(InputLabelsDataPath)

        if (len(listInputImagesFiles) != len(listInputLabelsFiles)):
            message = 'num Images \'%s\' and Labels \'%s\' not equal...' %(len(listInputImagesFiles), len(listInputLabelsFiles))
            catch_error_exception(message)

    if (args.isInputExtraLabels):
        InputExtraLabelsDataPath  = workDirsManager.getNameExistBaseDataPath(args.nameInputExtraLabelsRelPath)
        listInputExtraLabelsFiles = list_files_dir(InputExtraLabelsDataPath)

        if (len(listInputImagesFiles) != len(listInputExtraLabelsFiles)):
            message = 'num Images \'%s\' and Extra Labels \'%s\' not equal...' %(len(listInputImagesFiles), len(listInputExtraLabelsFiles))
            catch_error_exception(message)



    # Assign indexes for training / validation / testing data (randomly or with fixed order)
    if args.typedist == 'original' or args.typedist == 'random':
        sum_propData = sum(args.propData_trainvalidtest)
        if sum_propData != 1.0:
            message = 'Sum of props of Training / Validation / Testing data != 1.0 (%s)... Change input param...' %(sum_propData)
            catch_error_exception(message)

        numImagesFiles      = len(listInputImagesFiles)
        numfiles_training   = int(math.ceil(args.propData_trainvalidtest[0] * numImagesFiles))
        numfiles_validation = int(math.ceil(args.propData_trainvalidtest[1] * numImagesFiles))
        numfiles_testing    = max(0, numImagesFiles - numfiles_training - numfiles_validation)
        print("Num files for Training (%s)/ Validation (%s)/ Testing (%s)..." %(numfiles_training, numfiles_validation, numfiles_testing))

        if args.typedist == 'random':
            print("Distribute the Training / Validation / Testing data randomly...")
            indexesInputFiles = np.random.choice(range(numImagesFiles), size=numImagesFiles, replace=False)
        else:
            indexesInputFiles = range(numImagesFiles)

        indexesTrainingFiles   = indexesInputFiles[0:numfiles_training]
        indexesValidationFiles = indexesInputFiles[numfiles_training:numfiles_training+numfiles_validation]
        indexesTestingFiles    = indexesInputFiles[numfiles_training+numfiles_validation::]

    elif args.typedist == 'orderfile':
        args.infilevalidorder = args.infiletrainorder.replace('train','valid')
        args.infiletestorder  = args.infiletrainorder.replace('train','test')

        dictInputReferKeys = read_dictionary(InputReferKeysFile)
        listInputReferKeys = dictInputReferKeys.values()

        indexesTrainingFiles   = searchIndexesInputFilesFromReferKeysInFile(args.infiletrainorder, listInputReferKeys)
        indexesValidationFiles = searchIndexesInputFilesFromReferKeysInFile(args.infilevalidorder, listInputReferKeys)
        indexesTestingFiles    = searchIndexesInputFilesFromReferKeysInFile(args.infiletestorder,  listInputReferKeys)

        # Check whether there are files assigned to more than one set (Training / Validation / Testing)
        intersection_check = find_intersection_3lists(indexesTrainingFiles, indexesValidationFiles, indexesTestingFiles)
        if intersection_check != []:
            list_intersect_files = [dictInputReferKeys[basename(listInputImagesFiles[ind])] for ind in intersection_check]
            message = 'Found files assigned to more than one set (Training / Validation / Testing): %s...' %(list_intersect_files)
            catch_error_exception(message)



    # TRAINING DATA
    if indexesTrainingFiles == []:
        print("No Files assigned to Training Data:")
    else:
        print("Files assigned to Training Data:")
        for index in indexesTrainingFiles:
            input_image_file = listInputImagesFiles[index]
            output_image_file = join_path_names(TrainingDataPath, basename(input_image_file))
            print("%s --> %s" % (basename(output_image_file), input_image_file))
            makelink(input_image_file, output_image_file)

            if args.isPrepareLabels:
                input_label_file = listInputLabelsFiles[index]
                output_label_file = join_path_names(TrainingDataPath, basename(input_label_file))
                makelink(input_label_file, output_label_file)

            if args.isInputExtraLabels:
                input_extralabel_file = listInputExtraLabelsFiles[index]
                output_extralabel_file = join_path_names(TrainingDataPath, basename(input_extralabel_file))
                makelink(input_extralabel_file, output_extralabel_file)
        # endfor


    # VALIDATION DATA
    if indexesValidationFiles == []:
        print("No Files assigned to Validation Data:")
    else:
        print("Files assigned to Validation Data:")
        for index in indexesValidationFiles:
            input_image_file = listInputImagesFiles[index]
            output_image_file = join_path_names(ValidationDataPath, basename(input_image_file))
            print("%s --> %s" % (basename(output_image_file), input_image_file))
            makelink(input_image_file, output_image_file)

            if args.isPrepareLabels:
                input_label_file = listInputLabelsFiles[index]
                output_label_file = join_path_names(ValidationDataPath, basename(input_label_file))
                makelink(input_label_file, output_label_file)

            if args.isInputExtraLabels:
                input_extralabel_file = listInputExtraLabelsFiles[index]
                output_extralabel_file = join_path_names(ValidationDataPath, basename(input_extralabel_file))
                makelink(input_extralabel_file, output_extralabel_file)
        # endfor


    # TESTING DATA
    if indexesTestingFiles == []:
        print("No Files assigned to Testing Data:")
    else:
        print("Files assigned to Testing Data:")
        for index in indexesTestingFiles:
            input_image_file = listInputImagesFiles[index]
            output_image_file = join_path_names(TestingDataPath, basename(input_image_file))
            print("%s --> %s" % (basename(output_image_file), input_image_file))
            makelink(input_image_file, output_image_file)

            if args.isPrepareLabels:
                input_label_file = listInputLabelsFiles[index]
                output_label_file = join_path_names(TestingDataPath, basename(input_label_file))
                makelink(input_label_file, output_label_file)

            if args.isInputExtraLabels:
                input_extralabel_file = listInputExtraLabelsFiles[index]
                output_extralabel_file = join_path_names(TestingDataPath, basename(input_extralabel_file))
                makelink(input_extralabel_file, output_extralabel_file)
        # endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--nameInputImagesRelPath', type=str, default=NAME_PROCIMAGES_RELPATH)
    parser.add_argument('--nameInputLabelsRelPath', type=str, default=NAME_PROCLABELS_RELPATH)
    parser.add_argument('--nameInputReferKeysFile', type=str, default=NAME_REFERKEYSPROCIMAGE_FILE)
    parser.add_argument('--nameInputExtraLabelsRelPath', type=str, default=NAME_PROCEXTRALABELS_RELPATH)
    parser.add_argument('--nameTrainingDataRelPath', type=str, default=NAME_TRAININGDATA_RELPATH)
    parser.add_argument('--nameValidationDataRelPath', type=str, default=NAME_VALIDATIONDATA_RELPATH)
    parser.add_argument('--nameTestingDataRelPath', type=str, default=NAME_TESTINGDATA_RELPATH)
    parser.add_argument('--type', type=str, default='training')
    parser.add_argument('--typedist', type=str, default='original')
    parser.add_argument('--propData_trainvalidtest', type=str2tuple_float, default=PROPDATA_TRAINVALIDTEST)
    parser.add_argument('--infiletrainorder', type=str, default=None)
    args = parser.parse_args()

    if args.type == 'training':
        print("Distribute Training data: Processed Images and Labels...")
        args.isPrepareLabels    = True
        args.isInputExtraLabels = False

    elif args.type == 'testing':
        print("Prepare Testing data: Only Processed Images. Keep raw Images and Labels for testing...")
        args.isKeepRawImages      = True
        args.isPrepareLabels      = False
        args.isInputExtraLabels   = False
        args.isPrepareCentrelines = True
    else:
        message = 'Input param \'typedata\' = \'%s\' not valid, must be inside: \'%s\'...' % (args.type, LIST_TYPEDATA_AVAIL)
        catch_error_exception(message)

    if args.typedist not in LIST_TYPESDISTDATA_AVAIL:
        message = 'Input param \'typedistdata\' = \'%s\' not valid, must be inside: \'%s\'...' %(args.typedist, LIST_TYPESDISTDATA_AVAIL)
        catch_error_exception(message)

    if args.typedist == 'orderfile' and not args.infiletrainorder:
        message = 'Input \'infileorder\' file for \'fixed-order\' data distribution is needed...'
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
