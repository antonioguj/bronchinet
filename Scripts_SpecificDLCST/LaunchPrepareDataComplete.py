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
import traceback
import argparse

CODEDIR                    = '/home/antonio/Codes/Antonio_repository/AirwaySegmentation'
SCRIPT_EXTENDCROPPEDIMAGES = joinpathnames(CODEDIR, 'Scripts_SpecificDLCST/ExtendCroppedImagesToFullSize.py')
SCRIPT_RESCALEFACTORIMAGES = joinpathnames(CODEDIR, 'Scripts_ImageOperations/ComputeRescaleFactorImages.py')
SCRIPT_RESCALEROIMASKS     = joinpathnames(CODEDIR, 'Scripts_ImageOperations/ApplyOperationImages.py')
SCRIPT_BOUNDINGBOXIMAGES   = joinpathnames(CODEDIR, 'Scripts_ImageOperations/ComputeBoundingBoxImages.py')
SCRIPT_PREPAREDATA         = joinpathnames(CODEDIR, 'Scripts_Experiments/PrepareData.py')

CLUSTER_ARCHIVEDIR = 'agarcia@bigr-app001:/scratch/agarcia/Data/DLCST/'


def printCall(new_call):
    message = ' '.join(new_call)
    print("*" * 100)
    print("<<< Launch: %s >>>" %(message))
    print("*" * 100 +"\n")

def launchCall(new_call):
    Popen_obj = subprocess.Popen(new_call)
    Popen_obj.wait()



def main(args):

    SourceClusterDataDir = CLUSTER_ARCHIVEDIR

    nameSourceRawImagesPath   = joinpathnames(SourceClusterDataDir, 'CTs/')
    nameSourceRawLabelsPath   = joinpathnames(SourceClusterDataDir, 'Airways/')
    nameSourceRawRoiMasksPath = joinpathnames(SourceClusterDataDir, 'Lungs/')
    nameFoundBoundingBoxesFile= joinpathnames(SourceClusterDataDir, 'Others/found_boundingBox_croppedCTinFull.npy')

    OutputDataDir = makeUpdatedir(args.outputdatadir)
    # OutputDataDir = args.outputdatadir

    nameInputRawImagesPath   = joinpathnames(OutputDataDir, NAME_RAWIMAGES_RELPATH)
    nameInputRawLabelsPath   = joinpathnames(OutputDataDir, NAME_RAWLABELS_RELPATH)
    nameInputRawRoiMasksPath = joinpathnames(OutputDataDir, NAME_RAWROIMASKS_RELPATH)
    nameInputReferKeysPath   = joinpathnames(OutputDataDir, NAME_REFERKEYS_RELPATH)

    nameInputExtendedLabelsPath   = updatePathnameWithsuffix(nameInputRawLabelsPath, 'Extended')
    nameInputExtendedRoiMasksPath = updatePathnameWithsuffix(nameInputRawRoiMasksPath, 'Extended')


    list_calls_all = []


    #1st step: download and decompress data from the cluster
    new_call = ['rsync', '-avr', nameSourceRawImagesPath, nameInputRawImagesPath]
    list_calls_all.append(new_call)

    new_call = ['rsync', '-avr', nameSourceRawLabelsPath, nameInputRawLabelsPath]
    list_calls_all.append(new_call)

    new_call = ['rsync', '-avr', nameSourceRawRoiMasksPath, nameInputRawRoiMasksPath]
    list_calls_all.append(new_call)

    new_call = ['rsync', '-avr', nameFoundBoundingBoxesFile, OutputDataDir]
    list_calls_all.append(new_call)

    new_call = ['gunzip', '-vr', nameInputRawImagesPath]
    list_calls_all.append(new_call)

    new_call = ['gunzip', '-vr', nameInputRawLabelsPath]
    list_calls_all.append(new_call)

    new_call = ['gunzip', '-vr', nameInputRawRoiMasksPath]
    list_calls_all.append(new_call)


    # 2nd step: extend original labels and Roi masks, which are cropped, to full size
    new_call = ['python', SCRIPT_EXTENDCROPPEDIMAGES, nameInputRawLabelsPath, nameInputExtendedLabelsPath,
                '--datadir', args.outputdatadir]
    list_calls_all.append(new_call)

    new_call = ['python', SCRIPT_EXTENDCROPPEDIMAGES, nameInputRawRoiMasksPath, nameInputExtendedRoiMasksPath,
                '--datadir', args.outputdatadir]
    list_calls_all.append(new_call)


    # 3rd step: compute rescaling factors, and rescale the Roi masks to compute the bounding masks, if needed
    if args.rescaleImages:
        new_call = ['python', SCRIPT_RESCALEFACTORIMAGES,
                    '--datadir', args.outputdatadir]
        list_calls_all.append(new_call)


        nameInputRescaledRoiMasksPath = updatePathnameWithsuffix(nameInputExtendedRoiMasksPath, 'Recaled')
        nameRescaleFactorsFile        = joinpathnames(OutputDataDir, NAME_RESCALEFACTOR_FILE)

        new_call = ['python', SCRIPT_RESCALEROIMASKS, nameInputExtendedRoiMasksPath, nameInputRescaledRoiMasksPath,
                    '--type=rescale_mask',
                    '--rescalefile', nameRescaleFactorsFile,
                    '--referencedir', nameInputReferKeysPath]
        list_calls_all.append(new_call)


    # 4th step, compute the bounding-boxes, if needed
    if args.cropImages:
        if args.rescaleImages:
            new_call = ['python', SCRIPT_BOUNDINGBOXIMAGES,
                        '--datadir', args.outputdatadir,
                        '--nameInputRoiMasksRelPath', nameInputRescaledRoiMasksPath]
            list_calls_all.append(new_call)
        else:
            new_call = ['python', SCRIPT_BOUNDINGBOXIMAGES,
                        '--datadir', args.outputdatadir,
                        '--nameInputRoiMasksRelPath', nameInputExtendedRoiMasksPath]
            list_calls_all.append(new_call)


    # 5th step: prepare data
    new_call = ['python', SCRIPT_PREPAREDATA,
                '--datadir', args.outputdatadir,
                '--nameRawImagesRelPath', nameInputRawImagesPath,
                '--nameRawLabelsRelPath', nameInputExtendedLabelsPath,
                '--nameRawRoiMasksRelPath', nameInputExtendedRoiMasksPath,
                '--isPrepareLabels', str(args.isPrepareLabels),
                '--masksToRegionInterest', str(args.masksToRegionInterest),
                '--rescaleImages', str(args.rescaleImages),
                '--cropImages', str(args.cropImages)]
    list_calls_all.append(new_call)


    # 6th step: remove temporary data
    new_call = ['rm', '-r', nameInputRawImagesPath]
    list_calls_all.append(new_call)

    new_call = ['rm', '-r', nameInputRawLabelsPath]
    list_calls_all.append(new_call)

    new_call = ['rm', '-r', nameInputRawRoiMasksPath]
    list_calls_all.append(new_call)

    new_call = ['rm', '-r', nameInputExtendedLabelsPath]
    list_calls_all.append(new_call)

    new_call = ['rm', '-r', nameInputExtendedRoiMasksPath]
    list_calls_all.append(new_call)

    if args.rescaleImages:
        new_call = ['rm', '-r', nameInputRescaledRoiMasksPath]
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
    parser.add_argument('outputdatadir', type=str)
    parser.add_argument('--isPrepareLabels', type=str2bool, default=True)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)


