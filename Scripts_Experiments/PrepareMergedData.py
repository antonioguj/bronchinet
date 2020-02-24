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


TYPES_DUSTRIBUTE_DATA = ['original', 'random', 'fixedorder']


def main(args):
    # ---------- SETTINGS ----------
    nameTemplateOutputImagesFiles = 'images_proc-%0.2i.nii.gz'
    nameTemplateOutputLabelsFiles = 'labels_proc-%0.2i.nii.gz'
    nameTemplateOutputExtraLabelsFiles = 'cenlines_proc-%0.2i.nii.gz'
    # ---------- SETTINGS ----------



    listSourceImagesFiles_AllData = []
    listSourceLabelsFiles_AllData = []
    listSourceExtraLabelsFiles_AllData = []
    listDictSourceReferKeys_AllData = []

    for imerge_nameBaseDataPath in args.listMergeBaseDataPaths:
        # Check that source data exists
        if not isExistdir(imerge_nameBaseDataPath):
            message = 'Base Data dir: \'%s\' does not exist...' % (imerge_nameBaseDataPath)
            CatchErrorException(message)

        SourceImagesDataPath  = joinpathnames(imerge_nameBaseDataPath, args.nameProcImagesRelPath)
        nameReferenceKeysFile = joinpathnames(imerge_nameBaseDataPath, args.nameReferenceKeysFile)

        if not isExistdir(SourceImagesDataPath):
            message = 'Images Work Data dir: \'%s\' does not exist...' % (SourceImagesDataPath)
            CatchErrorException(message)
        if not isExistfile(nameReferenceKeysFile):
            message = 'Reference Keys file: \'%s\' does not exist...' % (nameReferenceKeysFile)
            CatchErrorException(message)

        listSourceImagesFiles    = findFilesDirAndCheck(SourceImagesDataPath)
        dict_sourceReferKeys_new = readDictionary(nameReferenceKeysFile)
        listSourceImagesFiles_AllData.append(listSourceImagesFiles)
        listDictSourceReferKeys_AllData.append(dict_sourceReferKeys_new)


        if args.isPrepareLabels:
            SourceLabelsDataPath = joinpathnames(imerge_nameBaseDataPath, args.nameProcLabelsRelPath)

            if not isExistdir(SourceLabelsDataPath):
                message = 'Labels Work Data dir: \'%s\' does not exist...' % (SourceLabelsDataPath)
                CatchErrorException(message)

            listSourceLabelsFiles = findFilesDirAndCheck(SourceLabelsDataPath)
            listSourceLabelsFiles_AllData.append(listSourceLabelsFiles)
        #endif

        if args.isInputExtraLabels:
            SourceExtraLabelsDataPath = joinpathnames(imerge_nameBaseDataPath, args.nameProcExtraLabelsRelPath)

            if not isExistdir(SourceExtraLabelsDataPath):
                message = 'Labels Work Data dir: \'%s\' does not exist...' % (SourceExtraLabelsDataPath)
                CatchErrorException(message)

            listSourceExtraLabelsFiles = findFilesDirAndCheck(SourceExtraLabelsDataPath)
            listSourceExtraLabelsFiles_AllData.append(listSourceExtraLabelsFiles)
        #endif
    #endfor



    # Assign indexes for merging the source data (randomly or with fixed order)
    if args.typedistdata == 'original' or \
        args.typedistdata == 'random':

        indexes_source_files = []

        for idata, i_listSourceImagesDataPath in enumerate(listSourceImagesFiles_AllData):
            indexes_files_newdata = [(idata, index_file) for index_file in range(len(i_listSourceImagesDataPath))]
            indexes_source_files += indexes_files_newdata
        # endfor

        if args.typedistdata == 'random':
            print("Distribute the merged data randomly...")

            num_source_files_total = len(indexes_source_files)
            random_indexes_sizeTotal = np.random.choice(range(num_source_files_total), size=num_source_files_total, replace=False)
            indexes_source_files = [indexes_source_files[index] for index in random_indexes_sizeTotal]

    elif args.typedistdata == 'fixedorder':

        indexes_source_files = []

        with open(args.infilefixedorder, 'r') as fin:
            for in_referkey_file in fin.readlines():
                in_referkey_file = in_referkey_file.replace('\r\n','')

                is_found = False
                for icount_data, i_dictSourceReferKeys in enumerate(listDictSourceReferKeys_AllData):
                    i_listSourceReferKeys = i_dictSourceReferKeys.values()
                    if in_referkey_file in i_listSourceReferKeys:
                        index_pos_referkey = i_listSourceReferKeys.index(in_referkey_file)
                        indexes_source_files.append((icount_data, index_pos_referkey))
                        is_found = True
                        break
                # endfor
                if not is_found:
                    list_allSourceReferKeys = sum([elem.values() for elem in listDictSourceReferKeys_AllData], [])
                    message = 'Reference Key: \'%s\' not found in list of Source Image Files: \'%s\'...' %(in_referkey_file,
                                                                                                           list_allSourceReferKeys)
                    CatchErrorException(message)
    # ------------------------------------------



    # Create new base dir with merged data
    HomeDir = dirnamepathdir(args.listMergeBaseDataPaths[0])
    DataDir = '+'.join(basenamedir(idir).split('_')[0] for idir in args.listMergeBaseDataPaths) +'_Processed'
    DataDir = joinpathnames(HomeDir, DataDir)

    OutputDataDir = makeupdatedir(DataDir)
    # OutputDataDir = DataDir

    workDirsManager = WorkDirsManager(OutputDataDir)
    OutputImagesDataPath = workDirsManager.getNameNewPath(args.nameProcImagesRelPath)

    if args.isPrepareLabels:
        OutputLabelsDataPath = workDirsManager.getNameNewPath(args.nameProcLabelsRelPath)

    if args.isInputExtraLabels:
        OutputExtraLabelsDataPath = workDirsManager.getNameNewPath(args.nameProcExtraLabelsRelPath)



    dict_referenceKeys = OrderedDict()

    for icount, (index_data, index_image_file) in enumerate(indexes_source_files):

        insource_image_file    = listSourceImagesFiles_AllData[index_data][index_image_file]
        insource_referkey_file = listDictSourceReferKeys_AllData[index_data][basename(insource_image_file)]
        output_image_file      = joinpathnames(OutputImagesDataPath, nameTemplateOutputImagesFiles % (icount+1))
        print("%s --> %s (%s)" % (basename(output_image_file), insource_image_file, basename(insource_referkey_file)))

        makelink(insource_image_file, output_image_file)

        # save this image in reference keys
        dict_referenceKeys[basename(output_image_file)] = basename(insource_referkey_file)


        if args.isPrepareLabels:
            insource_label_file = listSourceLabelsFiles_AllData[index_data][index_image_file]
            output_label_file   = joinpathnames(OutputLabelsDataPath, nameTemplateOutputLabelsFiles % (icount+1))

            makelink(insource_label_file, output_label_file)

        if args.isInputExtraLabels:
            insource_extralabel_file = listSourceExtraLabelsFiles_AllData[index_data][index_image_file]
            output_extralabel_file   = joinpathnames(OutputExtraLabelsDataPath, nameTemplateOutputExtraLabelsFiles % (icount+1))

            makelink(insource_extralabel_file, output_extralabel_file)
    #endfor


    # Save dictionary in file
    nameoutfile = joinpathnames(OutputDataDir, args.nameReferenceKeysFile)
    saveDictionary(nameoutfile, dict_referenceKeys)
    nameoutfile = joinpathnames(OutputDataDir, args.nameReferenceKeysFile.replace('.npy','.csv'))
    saveDictionary_csv(nameoutfile, dict_referenceKeys)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('listMergeBaseDataPaths', nargs='+', type=str, default=None)
    parser.add_argument('--nameProcImagesRelPath', type=str, default=NAME_PROCIMAGES_RELPATH)
    parser.add_argument('--nameProcLabelsRelPath', type=str, default=NAME_PROCLABELS_RELPATH)
    parser.add_argument('--nameProcExtraLabelsRelPath', type=str, default=NAME_PROCEXTRALABELS_RELPATH)
    parser.add_argument('--nameReferenceKeysFile', type=str, default=NAME_REFERENCEKEYS_FILE)
    parser.add_argument('--isPrepareLabels', type=str2bool, default=True)
    parser.add_argument('--isInputExtraLabels', type=str2bool, default=False)
    parser.add_argument('--typedistdata', type=str, default='original')
    parser.add_argument('--infilefixedorder', type=str, default=None)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in sorted(vars(args).iteritems()):
        print("\'%s\' = %s" %(key, value))

    if args.typedistdata not in TYPES_DUSTRIBUTE_DATA:
        message = 'Input for Type Distribute Data not valid: \'%s\'. Values accepted are: \'%s\'...' %(args.typedistdata,
                                                                                                       TYPES_DUSTRIBUTE_DATA)
        CatchErrorException(message)

    if args.typedistdata == 'fixedorder':
        if not args.infilefixedorder or \
            not isExistfile(args.infilefixedorder):
            message = 'File for Fixed order Distribution of Data not found: \'%s\'...' %(args.infilefixedorder)
            CatchErrorException(message)

    main(args)