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
from collections import OrderedDict
import argparse


def searchIndexesInputFilesFromReferKeysInFile(in_readfile, list_input_referKeys_allData):
    if not isExistfile(in_readfile):
        message = 'File for fixed-order distribution of data \'infileorder\' not found: \'%s\'...' % (in_readfile)
        CatchErrorException(message)

    out_indexes_input_files = []
    with open(in_readfile, 'r') as fin:
        for in_referkey_file in fin.readlines():
            in_referkey_file = in_referkey_file.replace('\r\n', '')

            is_found = False
            for icount_data, it_list_input_referKeys in enumerate(list_input_referKeys_allData):
                if in_referkey_file in it_list_input_referKeys:
                    index_pos_referkey_file = it_list_input_referKeys.index(in_referkey_file)
                    out_indexes_input_files.append((icount_data, index_pos_referkey_file))
                    is_found = True
                    break
            # endfor
            if not is_found:
                listAll_input_referKeys = sum(list_input_referKeys_allData,[])
                message = '\'%s\' not found in list of Input Reference Keys: \'%s\'...' % (in_referkey_file, listAll_input_referKeys)
                CatchErrorException(message)
    # --------------------------------------
    return out_indexes_input_files


TYPES_DUSTRIBUTE_DATA = ['original', 'random', 'orderfile']



def main(args):
    # ---------- SETTINGS ----------
    nameTemplateOutputImagesFiles = 'images_proc-%0.2i.nii.gz'
    nameTemplateOutputLabelsFiles = 'labels_proc-%0.2i.nii.gz'
    nameTemplateOutputExtraLabelsFiles = 'cenlines_proc-%0.2i.nii.gz'
    # ---------- SETTINGS ----------



    listInputImagesFiles_allData = []
    listInputLabelsFiles_allData = []
    listInputExtraLabelsFiles_allData = []
    listDictInputReferKeys_allData = []

    for imerge_nameDataPath in args.listMergeDataPaths:
        if not isExistdir(imerge_nameDataPath):
            message = 'Base Data dir: \'%s\' does not exist...' % (imerge_nameDataPath)
            CatchErrorException(message)

        workDirsManager    = WorkDirsManager(imerge_nameDataPath)
        InputImagesPath    = workDirsManager.getNameExistPath(args.nameInOutImagesRelPath)
        InputReferKeysFile = workDirsManager.getNameExistFile(args.nameInOutReferKeysFile)

        listInputImagesFiles = findFilesDirAndCheck(InputImagesPath)
        dictInputReferKeys   = readDictionary(InputReferKeysFile)
        listInputImagesFiles_allData.append(listInputImagesFiles)
        listDictInputReferKeys_allData.append(dictInputReferKeys)

        if args.isPrepareLabels:
            InputLabelsPath      = workDirsManager.getNameExistPath(args.nameInOutLabelsRelPath)
            listInputLabelsFiles = findFilesDirAndCheck(InputLabelsPath)
            listInputLabelsFiles_allData.append(listInputLabelsFiles)
        #endif

        if args.isInputExtraLabels:
            InputExtraLabelsPath      = workDirsManager.getNameExistPath(args.nameInOutExtraLabelsRelPath)
            listInputExtraLabelsFiles = findFilesDirAndCheck(InputExtraLabelsPath)
            listInputExtraLabelsFiles_allData.append(listInputExtraLabelsFiles)
        #endif
    #endfor



    # Assign indexes for merging the source data (randomly or with fixed order)
    if args.typedistdata == 'original' or args.typedistdata == 'random':
        indexesMergeInputFiles = []

        for idata, it_listInputImagesFiles in enumerate(listInputImagesFiles_allData):
            indexes_files_this = [(idata, index_file) for index_file in range(len(it_listInputImagesFiles))]
            indexesMergeInputFiles += indexes_files_this
        # endfor

        if args.typedistdata == 'random':
            print("Distribute the merged data randomly...")

            num_inputfiles_total = len(indexesMergeInputFiles)
            random_indexes_sizeTotal = np.random.choice(range(num_inputfiles_total), size=num_inputfiles_total, replace=False)
            indexesMergeInputFiles = [indexesMergeInputFiles[index] for index in random_indexes_sizeTotal]

    elif args.typedistdata == 'orderfile':
        listInputReferKeys_allData = [elem.values() for elem in listDictInputReferKeys_allData]

        indexesMergeInputFiles = searchIndexesInputFilesFromReferKeysInFile(args.infileorder, listInputReferKeys_allData)



    # Create new base dir with merged data
    HomeDir = dirnamepathdir(args.listMergeDataPaths[0])
    DataDir = '+'.join(basenamedir(idir).split('_')[0] for idir in args.listMergeDataPaths) +'_Processed'
    DataDir = joinpathnames(HomeDir, DataDir)

    OutputDataDir = makeUpdatedir(DataDir)
    # OutputDataDir = DataDir

    workDirsManager      = WorkDirsManager(OutputDataDir)
    OutputImagesDataPath = workDirsManager.getNameNewPath(args.nameInOutImagesRelPath)
    OutputReferKeysFile  = workDirsManager.getNameNewFile(args.nameInOutReferKeysFile)

    if args.isPrepareLabels:
        OutputLabelsDataPath = workDirsManager.getNameNewPath(args.nameInOutLabelsRelPath)

    if args.isInputExtraLabels:
        OutputExtraLabelsDataPath = workDirsManager.getNameNewPath(args.nameInOutExtraLabelsRelPath)



    out_dictReferenceKeys = OrderedDict()

    for icount, (index_data, index_image_file) in enumerate(indexesMergeInputFiles):

        input_image_file  = listInputImagesFiles_allData[index_data][index_image_file]
        in_referkey_file  = listDictInputReferKeys_allData[index_data][basename(input_image_file)]
        output_image_file = joinpathnames(OutputImagesDataPath, nameTemplateOutputImagesFiles % (icount+1))
        print("%s --> %s (%s)" % (basename(output_image_file), input_image_file, basename(in_referkey_file)))
        if args.isLinkmergedfiles:
            makelink(input_image_file, output_image_file)
        else:
            copyfile(input_image_file, output_image_file)

        # save this image in reference keys
        out_dictReferenceKeys[basename(output_image_file)] = basename(in_referkey_file)


        if args.isPrepareLabels:
            input_label_file  = listInputLabelsFiles_allData[index_data][index_image_file]
            output_label_file = joinpathnames(OutputLabelsDataPath, nameTemplateOutputLabelsFiles % (icount+1))
            if args.isLinkmergedfiles:
                makelink(input_label_file, output_label_file)
            else:
                copyfile(input_label_file, output_label_file)

        if args.isInputExtraLabels:
            input_extralabel_file  = listInputExtraLabelsFiles_allData[index_data][index_image_file]
            output_extralabel_file = joinpathnames(OutputExtraLabelsDataPath, nameTemplateOutputExtraLabelsFiles % (icount+1))
            if args.isLinkmergedfiles:
                makelink(input_extralabel_file, output_extralabel_file)
            else:
                copyfile(input_extralabel_file, output_extralabel_file)
    #endfor


    # Save dictionary in file
    saveDictionary(OutputReferKeysFile, out_dictReferenceKeys)
    saveDictionary_csv(OutputReferKeysFile.replace('.npy','.csv'), out_dictReferenceKeys)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('listMergeDataPaths', nargs='+', type=str, default=None)
    parser.add_argument('--nameInOutImagesRelPath', type=str, default=NAME_PROCIMAGES_RELPATH)
    parser.add_argument('--nameInOutLabelsRelPath', type=str, default=NAME_PROCLABELS_RELPATH)
    parser.add_argument('--nameInOutExtraLabelsRelPath', type=str, default=NAME_PROCEXTRALABELS_RELPATH)
    parser.add_argument('--nameInOutReferKeysFile', type=str, default=NAME_PROCREFERKEYS_FILE)
    parser.add_argument('--isPrepareLabels', type=str2bool, default=True)
    parser.add_argument('--isInputExtraLabels', type=str2bool, default=False)
    parser.add_argument('--typedistdata', type=str, default='original')
    parser.add_argument('--infileorder', type=str, default=None)
    parser.add_argument('--isLinkmergedfiles', type=str2bool, default=True)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in sorted(vars(args).iteritems()):
        print("\'%s\' = %s" %(key, value))

    if args.typedistdata not in TYPES_DUSTRIBUTE_DATA:
        message = 'Input for Type Distribute Data not valid: \'%s\'. Values accepted are: \'%s\'...' %(args.typedistdata,
                                                                                                       TYPES_DUSTRIBUTE_DATA)
        CatchErrorException(message)

    if args.typedistdata == 'orderfile' and not args.infileorder:
        message = 'Input for file for Fixed-order distribution of data \'infileorder\' needed...'
        CatchErrorException(message)

    main(args)