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
from CommonUtil.ErrorMessages import *
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import *
from Networks.Networks_NEW import *
from Postprocessing.SlidingWindowReconstructorImages import *
import argparse


def main(args):

    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameBaseDataPath()
    OrigImagesPath   = workDirsManager.getNameExistPath(BaseDataPath, 'RawImages')
    ComputeMasksPath = workDirsManager.getNameNewPath  (BaseDataPath, 'ProbNnetoutMasks')

    # Get the file list:
    nameImagesFiles   = '*.dcm'
    nameOutMasksFiles = lambda in_name: joinpathnames(ComputeMasksPath, filenamenoextension(in_name) + '_probnnetout.nii.gz')

    listImagesFiles = findFilesDir(OrigImagesPath, nameImagesFiles)
    nbImagesFiles   = len(listImagesFiles)


    # Retrieve training model
    modelConstructor = DICTAVAILNETWORKS3D(IMAGES_DIMS_Z_X_Y, args.model)
    modelConstructor.type_padding = 'valid'

    if args.size_out_nnet == None:
        args.size_out_nnet = modelConstructor.get_size_output_full_Unet()

    print("For input images of size: %s; Output of Neural Networks are images of size: %s..." %(IMAGES_DIMS_Z_X_Y, args.size_out_nnet))


    for imagesFile in listImagesFiles:

        print('\'%s\'...' % (imagesFile))

        images_array = FileReader.getImageArray(imagesFile)


        print("Compute masks proportion output...")

        if (args.slidingWindowImages):

            images_reconstructor = SlidingWindowReconstructorImages3D(IMAGES_DIMS_Z_X_Y, images_array.shape, args.prop_overlap_Z_X_Y, size_outnnet_sample=args.size_out_nnet)
        else:
            images_reconstructor = SlidingWindowReconstructorImages3D(IMAGES_DIMS_Z_X_Y, images_array.shape, (0.0, 0.0, 0.0), size_outnnet_sample=args.size_out_nnet)

        masks_probValidConvNnet_output_array = images_reconstructor.get_filtering_map_array()


        FileReader.writeImageArray(nameOutMasksFiles(imagesFile), masks_probValidConvNnet_output_array)
    #endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--model', default='Unet3D')
    parser.add_argument('--size_out_nnet', type=str2tuplefloat, default=IMAGES_SIZE_OUT_NNET)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)