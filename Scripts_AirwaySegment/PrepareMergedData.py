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
from collections import OrderedDict
import argparse


def searchIndexesInputFilesFromReferKeysInFile(in_readfile, list_input_referKeys_allData):
    if not is_exist_file(in_readfile):
        message = 'File for fixed-order distribution of data \'infileorder\' not found: \'%s\'...' % (in_readfile)
        catch_error_exception(message)

    out_indexes_input_files = []
    with open(in_readfile, 'r') as fin:
        for in_referkey_file in fin.readlines():
            in_referkey_file = in_referkey_file.replace('\n','').replace('\r','')

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
                catch_error_exception(message)
    # --------------------------------------
    return out_indexes_input_files


LIST_TYPEDATA_AVAIL = ['training', 'testing']
LIST_TYPESDISTDATA_AVAIL = ['original', 'random', 'orderfile']



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
        if not is_exist_dir(imerge_nameDataPath):
            message = 'Base Data dir: \'%s\' does not exist...' % (imerge_nameDataPath)
            catch_error_exception(message)

        workDirsManager    = GeneralDirManager(imerge_nameDataPath)
        InputImagesPath    = workDirsManager.get_pathdir_exist(args.nameInOutImagesRelPath)
        InputReferKeysFile = workDirsManager.get_pathfile_exist(args.nameInOutReferKeysFile)

        listInputImagesFiles = list_files_dir(InputImagesPath)
        dictInputReferKeys   = read_dictionary(InputReferKeysFile)
        listInputImagesFiles_allData.append(listInputImagesFiles)
        listDictInputReferKeys_allData.append(dictInputReferKeys)

        if args.isPrepareLabels:
            InputLabelsPath      = workDirsManager.get_pathdir_exist(args.nameInOutLabelsRelPath)
            listInputLabelsFiles = list_files_dir(InputLabelsPath)
            listInputLabelsFiles_allData.append(listInputLabelsFiles)
        #endif

        if args.isInputExtraLabels:
            InputExtraLabelsPath      = workDirsManager.get_pathdir_exist(args.nameInOutExtraLabelsRelPath)
            listInputExtraLabelsFiles = list_files_dir(InputExtraLabelsPath)
            listInputExtraLabelsFiles_allData.append(listInputExtraLabelsFiles)
        #endif
    #endfor



    # Assign indexes for merging the source data (randomly or with fixed order)
    if args.typedist == 'original' or args.typedist == 'random':
        indexesMergeInputFiles = []

        for idata, it_listInputImagesFiles in enumerate(listInputImagesFiles_allData):
            indexes_files_this = [(idata, index_file) for index_file in range(len(it_listInputImagesFiles))]
            indexesMergeInputFiles += indexes_files_this
        # endfor

        if args.typedist == 'random':
            print("Distribute the merged data randomly...")

            num_inputfiles_total = len(indexesMergeInputFiles)
            random_indexes_sizeTotal = np.random.choice(range(num_inputfiles_total), size=num_inputfiles_total, replace=False)
            indexesMergeInputFiles = [indexesMergeInputFiles[index] for index in random_indexes_sizeTotal]

    elif args.typedist == 'orderfile':
        listInputReferKeys_allData = [elem.values() for elem in listDictInputReferKeys_allData]

        indexesMergeInputFiles = searchIndexesInputFilesFromReferKeysInFile(args.infileorder, listInputReferKeys_allData)



    # Create new base dir with merged data
    HomeDir = dirname_dir(args.listMergeDataPaths[0])
    DataDir = '+'.join(basename_dir(idir).split('_')[0] for idir in args.listMergeDataPaths) + '_Processed'
    DataDir = join_path_names(HomeDir, DataDir)

    OutputDataDir = makedir(update_dirname(DataDir))
    # OutputDataDir = DataDir

    workDirsManager      = GeneralDirManager(OutputDataDir)
    OutputImagesDataPath = workDirsManager.get_pathdir_new(args.nameInOutImagesRelPath)
    OutputReferKeysFile  = workDirsManager.get_pathfile_new(args.nameInOutReferKeysFile)

    if args.isPrepareLabels:
        OutputLabelsDataPath = workDirsManager.get_pathdir_new(args.nameInOutLabelsRelPath)

    if args.isInputExtraLabels:
        OutputExtraLabelsDataPath = workDirsManager.get_pathdir_new(args.nameInOutExtraLabelsRelPath)



    outdict_referenceKeys = OrderedDict()

    for icount, (index_data, index_image_file) in enumerate(indexesMergeInputFiles):

        input_image_file  = listInputImagesFiles_allData[index_data][index_image_file]
        in_referkey_file  = listDictInputReferKeys_allData[index_data][basename_file_noext(input_image_file)]
        output_image_file = join_path_names(OutputImagesDataPath, nameTemplateOutputImagesFiles % (icount + 1))
        print("%s --> %s (%s)" % (basename(output_image_file), input_image_file, basename(in_referkey_file)))
        if args.isLinkmergedfiles:
            makelink(input_image_file, output_image_file)
        else:
            copyfile(input_image_file, output_image_file)

        # save this image in reference keys
        outdict_referenceKeys[basename_file_noext(output_image_file)] = basename(in_referkey_file)


        if args.isPrepareLabels:
            input_label_file  = listInputLabelsFiles_allData[index_data][index_image_file]
            output_label_file = join_path_names(OutputLabelsDataPath, nameTemplateOutputLabelsFiles % (icount + 1))
            if args.isLinkmergedfiles:
                makelink(input_label_file, output_label_file)
            else:
                copyfile(input_label_file, output_label_file)

        if args.isInputExtraLabels:
            input_extralabel_file  = listInputExtraLabelsFiles_allData[index_data][index_image_file]
            output_extralabel_file = join_path_names(OutputExtraLabelsDataPath, nameTemplateOutputExtraLabelsFiles % (icount + 1))
            if args.isLinkmergedfiles:
                makelink(input_extralabel_file, output_extralabel_file)
            else:
                copyfile(input_extralabel_file, output_extralabel_file)
    #endfor


    # Save dictionary in file
    save_dictionary(OutputReferKeysFile, outdict_referenceKeys)
    save_dictionary_csv(OutputReferKeysFile.replace('.npy', '.csv'), outdict_referenceKeys)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('listMergeDataPaths', nargs='+', type=str, default=None)
    parser.add_argument('--nameInOutImagesRelPath', type=str, default=NAME_PROCIMAGES_RELPATH)
    parser.add_argument('--nameInOutLabelsRelPath', type=str, default=NAME_PROCLABELS_RELPATH)
    parser.add_argument('--nameInOutExtraLabelsRelPath', type=str, default=NAME_PROCEXTRALABELS_RELPATH)
    parser.add_argument('--nameInOutReferKeysFile', type=str, default=NAME_REFERKEYSPROCIMAGE_FILE)
    parser.add_argument('--type', type=str, default='training')
    parser.add_argument('--typedist', type=str, default='original')
    parser.add_argument('--infileorder', type=str, default=None)
    parser.add_argument('--isLinkmergedfiles', type=str2bool, default=True)
    args = parser.parse_args()

    if args.type == 'training':
        print("Distribute Training data: Processed Images and Labels...")
        args.isPrepareLabels    = True
        args.isInputExtraLabels = False

    elif args.type == 'testing':
        print("Prepare Testing data: Only Processed Images...")
        args.isPrepareLabels      = False
        args.isInputExtraLabels   = False
    else:
        message = 'Input param \'typedata\' = \'%s\' not valid, must be inside: \'%s\'...' % (args.type, LIST_TYPEDATA_AVAIL)
        catch_error_exception(message)

    if args.typedist not in LIST_TYPESDISTDATA_AVAIL:
        message = 'Input param \'typedistdata\' = \'%s\' not valid, must be inside: \'%s\'...' %(args.typedist, LIST_TYPESDISTDATA_AVAIL)
        catch_error_exception(message)

    if args.typedist == 'orderfile' and not args.infileorder:
        message = 'Input \'infileorder\' file for \'fixed-order\' data distribution is needed...'
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in sorted(vars(args).items()):
        print("\'%s\' = %s" %(key, value))

    main(args)