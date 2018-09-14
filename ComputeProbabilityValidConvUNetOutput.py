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
from Preprocessing.OperationsImages import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    nameRawImagesRelPath    = 'RawImages'
    nameComputeMasksRelPath = 'ProbNnetoutMasks'

    # Get the file list:
    nameImagesFiles   = '*.dcm'
    nameOutMasksFiles = lambda in_name: filenamenoextension(in_name) + '_probnnetout.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameBaseDataPath()
    InputImagesPath  = workDirsManager.getNameExistPath(BaseDataPath, nameRawImagesRelPath   )
    ComputeMasksPath = workDirsManager.getNameNewPath  (BaseDataPath, nameComputeMasksRelPath)

    listImagesFiles = findFilesDir(InputImagesPath, nameImagesFiles)
    nbImagesFiles   = len(listImagesFiles)


    # Retrieve training model
    modelConstructor = DICTAVAILNETWORKS3D(IMAGES_DIMS_Z_X_Y, args.model)
    modelConstructor.type_padding = 'valid'

    if args.size_out_nnet == None:
        args.size_out_nnet = modelConstructor.get_size_output_full_Unet()

    print("For input images of size: %s; Output of Neural Networks are images of size: %s..." %(IMAGES_DIMS_Z_X_Y, args.size_out_nnet))



    for images_file in listImagesFiles:

        print('\'%s\'...' % (images_file))

        images_array = FileReader.getImageArray(images_file)

        if (args.invertImageAxial):
            images_array = FlippingImages.compute(images_array, axis=0)


        print("Compute masks proportion output...")

        if (args.slidingWindowImages):

            images_reconstructor = SlidingWindowReconstructorImages3D(IMAGES_DIMS_Z_X_Y, images_array.shape, args.prop_overlap_Z_X_Y, size_outnnet_sample=args.size_out_nnet)
        else:
            images_reconstructor = SlidingWindowReconstructorImages3D(IMAGES_DIMS_Z_X_Y, images_array.shape, (0.0, 0.0, 0.0), size_outnnet_sample=args.size_out_nnet)

        masks_probValidConvNnet_output_array = images_reconstructor.get_filtering_map_array()


        out_masksFilename = joinpathnames(ComputeMasksPath, nameOutMasksFiles(images_file))

        FileReader.writeImageArray(out_masksFilename, masks_probValidConvNnet_output_array)
    #endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--model', default='Unet3D')
    parser.add_argument('--invertImageAxial', type=str2bool, default=INVERTIMAGEAXIAL)
    parser.add_argument('--size_out_nnet', type=str2tuplefloat, default=IMAGES_SIZE_OUT_NNET)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)