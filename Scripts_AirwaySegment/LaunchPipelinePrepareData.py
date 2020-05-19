#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
#######################################################################################

from Common.Constants import *
from Common.FunctionsUtil import *
from Common.WorkDirsManager import *
import subprocess
import argparse


CODEDIR                     = '/home/antonio/Codes/Antonio_repository/AirwaySegmentation/'
SCRIPT_CONVERTTONIFTY       = joinpathnames(CODEDIR, 'Scripts_Util/ConvertImageToNifty.py')
SCRIPT_BINARISEMASKS        = joinpathnames(CODEDIR, 'Scripts_Util/ApplyOperationImages.py')
SCRIPT_GETTRACHEAMAINBRONCHI= joinpathnames(CODEDIR, 'Scripts_Util/ApplyOperationImages.py')
SCRIPT_COMPUTECENTRELINES   = joinpathnames(CODEDIR, 'Scripts_Util/ApplyOperationImages.py')
SCRIPT_RESCALEROIMASKS      = joinpathnames(CODEDIR, 'Scripts_Util/ApplyOperationImages.py')
SCRIPT_EXTENDCROPPEDIMAGES  = joinpathnames(CODEDIR, 'Scripts_Util/SpecificDLCST/ExtendCroppedImagesToFullSize.py')
SCRIPT_RESCALEFACTORIMAGES  = joinpathnames(CODEDIR, 'Scripts_AirwaySegment/ComputeRescaleFactorImages.py')
SCRIPT_BOUNDINGBOXIMAGES    = joinpathnames(CODEDIR, 'Scripts_AirwaySegment/ComputeBoundingBoxImages.py')
SCRIPT_PREPAREDATA          = joinpathnames(CODEDIR, 'Scripts_Experiments/PrepareData.py')

CLUSTER_ARCHIVEDIR = 'agarcia@bigr-app001:/scratch/agarcia/Data/'

LIST_TYPEDATA_AVAIL = ['training', 'testing']


def printCall(new_call):
    message = ' '.join(new_call)
    print("*" * 100)
    print("<<< Launch: %s >>>" %(message))
    print("*" * 100 +"\n")

def launchCall(new_call):
    Popen_obj = subprocess.Popen(new_call)
    Popen_obj.wait()


def create_task_replace_dirs(input_dir, input_dir_to_replace):
    new_call_1 = ['rm', '-r', input_dir]
    new_call_2 = ['mv', input_dir_to_replace, input_dir]
    return [new_call_1, new_call_2]


def create_task_decompress_data(input_data_dir, is_keep_files):
    list_files = findFilesDir(input_data_dir)
    extension_file = fileextension(list_files[0])
    sublist_calls = []

    if extension_file == '.dcm.gz':
        # decompress data
        new_call = ['gunzip', '-vr', input_data_dir]
        sublist_calls.append(new_call)

    if is_keep_files and extension_file in ['.dcm', '.dcm.gz']:
        # convert to nifty, if we keep the raw images for testing
        new_input_data_dir = updatePathnameWithsuffix(input_data_dir, 'Nifty')
        new_call = ['python', SCRIPT_CONVERTTONIFTY, input_data_dir, new_input_data_dir]
        sublist_calls.append(new_call)

        # replace output folder with nifty files
        new_sublist_calls = create_task_replace_dirs(input_data_dir, new_input_data_dir)
        sublist_calls += new_sublist_calls

    return sublist_calls



def main(args):

    SourceClusterDataDir = joinpathnames(CLUSTER_ARCHIVEDIR, args.inclustercasedir)

    nameSourceRawImagesPath   = joinpathnames(SourceClusterDataDir, 'CTs/')
    nameSourceRawLabelsPath   = joinpathnames(SourceClusterDataDir, 'Airways/')
    nameSourceRawRoiMasksPath = joinpathnames(SourceClusterDataDir, 'Lungs/')
    if args.isPrepareCoarseAirways:
        nameSourceRawCoarseAirwaysPath= joinpathnames(SourceClusterDataDir, 'CoarseAirways/')
    if args.inclustercasedir in ['DLCST', 'DLCST/']:
        nameSourceFoundBoundBoxesFile = joinpathnames(SourceClusterDataDir, 'Others/found_boundingBox_croppedCTinFull.npy')


    # OutputDataDir = makeUpdatedir(args.outputdatadir)
    OutputDataDir = args.outputdatadir
    makedir(OutputDataDir)

    OutputDataDir            = joinpathnames(currentdir(), OutputDataDir)
    nameInputRawImagesPath   = joinpathnames(OutputDataDir, NAME_RAWIMAGES_RELPATH)
    nameInputRawLabelsPath   = joinpathnames(OutputDataDir, NAME_RAWLABELS_RELPATH)
    nameInputRawRoiMasksPath = joinpathnames(OutputDataDir, NAME_RAWROIMASKS_RELPATH)
    nameInputReferKeysPath   = joinpathnames(OutputDataDir, NAME_REFERKEYS_RELPATH)
    if args.isPrepareCentrelines:
        nameInputRawCentrelinesPath   = joinpathnames(OutputDataDir, NAME_RAWCENTRELINES_RELPATH)
    if args.isPrepareCoarseAirways:
        nameInputRawCoarseAirwaysPath = joinpathnames(OutputDataDir, NAME_RAWCOARSEAIRWAYS_RELPATH)
    if args.rescaleImages:
        nameInputRescaleFactorsFile   = joinpathnames(OutputDataDir, NAME_RESCALEFACTOR_FILE)
    if args.inclustercasedir in ['DLCST', 'DLCST/']:
        nameInputFoundBoundBoxesFile  = joinpathnames(OutputDataDir, basename(nameSourceFoundBoundBoxesFile))



    list_calls_all = []

    # 1st: download data from the cluster
    new_call = ['rsync', '-avr', nameSourceRawImagesPath, nameInputRawImagesPath]
    list_calls_all.append(new_call)

    new_call = ['rsync', '-avr', nameSourceRawLabelsPath, nameInputRawLabelsPath]
    list_calls_all.append(new_call)

    new_call = ['rsync', '-avr', nameSourceRawRoiMasksPath, nameInputRawRoiMasksPath]
    list_calls_all.append(new_call)

    if args.isPrepareCoarseAirways:
        new_call = ['rsync', '-avr', nameSourceRawCoarseAirwaysPath, nameInputRawCoarseAirwaysPath]
        list_calls_all.append(new_call)

    if args.inclustercasedir in ['DLCST', 'DLCST/']:
        new_call = ['rsync', '-avr', nameSourceFoundBoundBoxesFile, nameInputFoundBoundBoxesFile]
        list_calls_all.append(new_call)

    # Iterate over the list and carry out call serially
    for icall in list_calls_all:
        printCall(icall)
        try:
            launchCall(icall)
        except Exception as ex:
            traceback.print_exc(file=sys.stdout)
            message = 'Call failed. Stop pipeline...'
            CatchErrorException(message)
        print('\n')
    #endfor



    list_calls_all = []

    # 2nd: decompress (if needed) and prepare the downloaded data
    sublist_calls = create_task_decompress_data(nameInputRawImagesPath, args.isKeepRawImages)
    list_calls_all += sublist_calls

    sublist_calls = create_task_decompress_data(nameInputRawLabelsPath, args.isKeepRawImages)
    list_calls_all += sublist_calls

    sublist_calls = create_task_decompress_data(nameInputRawRoiMasksPath, args.isKeepRawImages)
    list_calls_all += sublist_calls

    if args.isPrepareCoarseAirways:
        sublist_calls = create_task_decompress_data(nameInputRawCoarseAirwaysPath, args.isKeepRawImages)
        list_calls_all += sublist_calls


    # binarise the input arrays for airway and lungs
    if args.isKeepRawImages:
        nameTempoBinaryLabelsPath   = updatePathnameWithsuffix(nameInputRawLabelsPath,   'Binary')
        nameTempoBinaryRoiMasksPath = updatePathnameWithsuffix(nameInputRawRoiMasksPath, 'Binary')

        new_call = ['python', SCRIPT_BINARISEMASKS, nameInputRawLabelsPath, nameTempoBinaryLabelsPath,
                    '--type', 'binarise']
        list_calls_all.append(new_call)

        new_call = ['python', SCRIPT_BINARISEMASKS, nameInputRawRoiMasksPath, nameTempoBinaryRoiMasksPath,
                    '--type', 'binarise']
        list_calls_all.append(new_call)

        # replace output folder with binarised masks
        new_sublist_calls = create_task_replace_dirs(nameInputRawLabelsPath, nameTempoBinaryLabelsPath)
        list_calls_all += new_sublist_calls

        new_sublist_calls = create_task_replace_dirs(nameInputRawRoiMasksPath, nameTempoBinaryRoiMasksPath)
        list_calls_all += new_sublist_calls

    # extract the labels for trachea and main bronchii from the coarse airways
    if args.isPrepareCoarseAirways:
        nameTempoTracheaMainBronchiPath = updatePathnameWithsuffix(nameInputRawCoarseAirwaysPath, 'TracheaMainBronchi')

        new_call = ['python', SCRIPT_GETTRACHEAMAINBRONCHI, nameInputRawCoarseAirwaysPath, nameTempoTracheaMainBronchiPath,
                    '--type', 'masklabels',
                    '--inmasklabels', '2', '3', '4',
                    '--nosuffixoutname', 'True']
        list_calls_all.append(new_call)

        # replace output folder with trachea / main bronchi masks
        new_sublist_calls = create_task_replace_dirs(nameInputRawCoarseAirwaysPath, nameTempoTracheaMainBronchiPath)
        list_calls_all += new_sublist_calls



    # 3rd: for DLCST data: extend the raw images from the cropped and flipped format found in the cluster
    if args.inclustercasedir in ['DLCST', 'DLCST/']:
        nameTempoExtendedLabelsPath   = updatePathnameWithsuffix(nameInputRawLabelsPath,   'Extended')
        nameTempoExtendedRoiMasksPath = updatePathnameWithsuffix(nameInputRawRoiMasksPath, 'Extended')

        new_call = ['python', SCRIPT_EXTENDCROPPEDIMAGES, nameInputRawLabelsPath, nameTempoExtendedLabelsPath,
                    '--referkeysdir', nameInputReferKeysPath,
                    '--inputBoundBoxesFile', nameInputFoundBoundBoxesFile]
        list_calls_all.append(new_call)

        new_call = ['python', SCRIPT_EXTENDCROPPEDIMAGES, nameInputRawRoiMasksPath, nameTempoExtendedRoiMasksPath,
                    '--referkeysdir', nameInputReferKeysPath,
                    '--inputBoundBoxesFile', nameInputFoundBoundBoxesFile]
        list_calls_all.append(new_call)

        # replace output folder with extended images
        sublist_calls = create_task_replace_dirs(nameInputRawLabelsPath, nameTempoExtendedLabelsPath)
        list_calls_all += sublist_calls

        sublist_calls = create_task_replace_dirs(nameInputRawRoiMasksPath, nameTempoExtendedRoiMasksPath)
        list_calls_all += sublist_calls

        if args.isPrepareCoarseAirways:
            nameTempoExtendedCoarseAirwaysPath = updatePathnameWithsuffix(nameInputRawCoarseAirwaysPath, 'Extended')

            new_call = ['python', SCRIPT_EXTENDCROPPEDIMAGES, nameInputRawCoarseAirwaysPath, nameTempoExtendedCoarseAirwaysPath,
                        '--referkeysdir', nameInputReferKeysPath,
                        '--inputBoundBoxesFile', nameInputFoundBoundBoxesFile]
            list_calls_all.append(new_call)

            # replace output folder with extended images
            new_sublist_calls = create_task_replace_dirs(nameInputRawCoarseAirwaysPath, nameTempoExtendedCoarseAirwaysPath)
            list_calls_all += new_sublist_calls



    # 4th: compute the ground-truth centrelines by thinning the ground-truth airways
    if args.isPrepareCentrelines:
        new_call = ['python', SCRIPT_COMPUTECENTRELINES, nameInputRawLabelsPath, nameInputRawCentrelinesPath,
                    '--type', 'thinning']
        list_calls_all.append(new_call)



    # 5th: compute rescaling factors, and rescale the Roi masks to compute the bounding masks
    if args.rescaleImages:
        nameTempoRescaledRoiMasksPath = updatePathnameWithsuffix(nameInputRawRoiMasksPath, 'Rescaled')

        new_call = ['python', SCRIPT_RESCALEFACTORIMAGES,
                    '--datadir', OutputDataDir,
                    '--fixedRescaleRes', str(args.fixedRescaleRes)]
        list_calls_all.append(new_call)

        new_call = ['python', SCRIPT_RESCALEROIMASKS, nameInputRawRoiMasksPath, nameTempoRescaledRoiMasksPath,
                    '--type', 'rescale_mask',
                    '--rescalefile', nameInputRescaleFactorsFile,
                    '--referencedir', nameInputReferKeysPath]
        list_calls_all.append(new_call)

        # replace output folder with rescaled Roi masks
        sublist_calls = create_task_replace_dirs(nameInputRawLabelsPath, nameTempoRescaledRoiMasksPath)
        list_calls_all += sublist_calls



    # 6th: compute the bounding-boxes around the Roi masks
    if args.cropImages:
        new_call = ['python', SCRIPT_BOUNDINGBOXIMAGES,
                    '--datadir', OutputDataDir,
                    '--isTwoBoundingBoxEachLungs', str(args.isTwoBoundingBoxEachLungs),
                    '--sizeBufferInBorders', str(args.sizeBufferInBorders),
                    '--sizeTrainImages', str(args.sizeTrainImages),
                    '--isSameSizeBoundBoxAllImages', str(args.isSameSizeBoundBoxAllImages),
                    '--fixedSizeBoundingBox', str(args.fixedSizeBoundingBox)]
        list_calls_all.append(new_call)



    # 7th: prepare the data
    new_call = ['python', SCRIPT_PREPAREDATA,
                '--datadir', OutputDataDir,
                '--isPrepareLabels', str(args.isPrepareLabels),
                '--isInputExtraLabels', 'False',
                '--isBinaryTrainMasks', 'True',
                '--masksToRegionInterest', str(args.masksToRegionInterest),
                '--rescaleImages', str(args.rescaleImages),
                '--cropImages', str(args.cropImages),
                '--isROIlabelsMultiROImasks', str(args.isTwoBoundingBoxEachLungs)]
    list_calls_all.append(new_call)



    # remove all the data not needed anymore
    if args.type == 'training':
        new_call = ['rm', '-r', nameInputRawImagesPath]
        list_calls_all.append(new_call)

        new_call = ['rm', '-r', nameInputRawLabelsPath]
        list_calls_all.append(new_call)

        new_call = ['rm', '-r', nameInputRawRoiMasksPath]
        list_calls_all.append(new_call)

        if args.inclustercasedir in ['DLCST', 'DLCST/']:
            new_call = ['rm', nameInputFoundBoundBoxesFile]
            list_calls_all.append(new_call)



    # Iterate over the list and carry out call serially
    for icall in list_calls_all:
        printCall(icall)
        try:
            launchCall(icall)
        except Exception as ex:
            traceback.print_exc(file=sys.stdout)
            message = 'Call failed. Stop pipeline...'
            CatchErrorException(message)
        print('\n')
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inclustercasedir', type=str)
    parser.add_argument('outputdatadir', type=str)
    parser.add_argument('--type', type=str, default='training')
    parser.add_argument('--sizeTrainImages', type=str2tupleint, default=IMAGES_DIMS_Z_X_Y)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    parser.add_argument('--fixedRescaleRes', type=str2tuplefloatOrNone, default=FIXEDRESCALERES)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--isTwoBoundingBoxEachLungs', type=str2bool, default=ISTWOBOUNDINGBOXEACHLUNGS)
    parser.add_argument('--fixedSizeBoundingBox', type=str2tupleintOrNone, default=FIXEDSIZEBOUNDINGBOX)
    args = parser.parse_args()

    if args.type == 'training':
        print("Prepare Training data: Processed Images and Labels...")
        args.isKeepRawImages       = False
        args.isPrepareLabels       = True
        args.isPrepareCentrelines  = False
        args.isPrepareCoarseAirways= False
        if args.cropImages:
            if args.isTwoBoundingBoxEachLungs:
                args.sizeBufferInBorders = (0, 0, 0)
                args.isSameSizeBoundBoxAllImages = True
                args.fixedSizeBoundingBox = args.sizeTrainImages
            else:
                args.sizeBufferInBorders = (20, 20, 20)
                args.isSameSizeBoundBoxAllImages = False
                args.fixedSizeBoundingBox = None

    elif args.type == 'testing':
        print("Prepare Testing data: Only Processed Images. Keep raw Images and Labels for testing...")
        args.isKeepRawImages       = True
        args.isPrepareLabels       = False
        args.isPrepareCentrelines  = True
        args.isPrepareCoarseAirways= True
        if args.cropImages:
            args.sizeBufferInBorders = (50, 50, 50)
            args.isSameSizeBoundBoxAllImages = False
            args.fixedSizeBoundingBox = None
    else:
        message = 'Input param \'typedata\' = \'%s\' not valid, must be inside: \'%s\'...' % (args.type, LIST_TYPEDATA_AVAIL)
        CatchErrorException(message)

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
