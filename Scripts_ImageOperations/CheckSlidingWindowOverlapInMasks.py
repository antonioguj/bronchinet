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
from Common.ErrorMessages import *
from Common.FunctionsUtil import *
from Common.WorkDirsManager import *
from DataLoaders.FileReaders import *
from Networks_Pytorch.Networks_WithGNN import *
from Postprocessing.ImageReconstructorManager import *
from Preprocessing.OperationImages import *
from Preprocessing.OperationMasks import *
import numpy as np
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameInputProcDataRelPath = 'ImagesWorkData/'
    nameInputReferMasksRelPath = 'Airways/'
    nameReferenceImgRelPath = 'Images/'
    nameOutputFilesRelPath = 'Test_slidingWindowOverlap_Full/'
    nameCropBoundingBoxes = 'cropBoundingBoxes_images.npy'

    def nameOutputFiles(namecase, size_image, slidewin_propOverlap):
        str_size_image = 'x'.join([str(s) for s in size_image])
        slidewin_propOverlap = 'x'.join([str(s).replace('.','') for s in slidewin_propOverlap])
        out_name = 'test_%s_sizeimg%s_propoverlap%s.nii.gz' %(namecase, str_size_image, slidewin_propOverlap)
        return out_name
    # ---------- SETTINGS ----------


    workDirsManager     = WorkDirsManager(args.basedir)
    BaseDataPath        = workDirsManager.getNameBaseDataPath()
    InputProcDataPath   = workDirsManager.getNameExistPath(BaseDataPath, nameInputProcDataRelPath)
    InputReferMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputReferMasksRelPath)
    ReferenceImgPath    = workDirsManager.getNameExistPath(BaseDataPath, nameReferenceImgRelPath)
    OutputFilesPath     = workDirsManager.getNameNewPath  (args.basedir, nameOutputFilesRelPath)

    listInputProcDataFiles  = findFilesDirAndCheck(InputProcDataPath)
    listInputReferMasksFiles= findFilesDirAndCheck(InputReferMasksPath)
    listReferenceImgsFiles  = findFilesDirAndCheck(ReferenceImgPath)

    if (args.cropImages):
        cropBoundingBoxesFileName = joinpathnames(args.datadir, nameCropBoundingBoxes)
        dict_cropBoundingBoxes = readDictionary(cropBoundingBoxesFileName)


    model_net = DICTAVAILMODELS('Unet', args.size_in_images,
                                nlevel=args.num_layers, nfeat=8,
                                isUse_valid_convs=args.isValidConvolutions)

    size_output_modelnet = tuple(model_net.get_size_output()[1:])
    if args.isValidConvolutions:
        print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..." % (str(args.size_in_images),
                                                                                               str(size_output_modelnet)))

    images_reconstructor = getImagesReconstructor3D(args.size_in_images,
                                                    args.slidingWindowImages,
                                                    args.slidewin_propOverlap,
                                                    use_TransformationImages=False,
                                                    isUse_valid_convs=args.isValidConvolutions,
                                                    size_output_model=size_output_modelnet,
                                                    isfilter_valid_outUnet=args.filterPredictProbMaps,
                                                    prop_valid_outUnet=args.prop_valid_outUnet)



    for ifile, in_procdata_file in enumerate(listInputProcDataFiles):
        print("\nInput: \'%s\'..." % (basename(in_procdata_file)))
        # Assign original images and masks files
        index_refer_img = getIndexOriginImagesFile(basename(in_procdata_file), beginString='images', firstIndex='01')
        in_refermask_file = listInputReferMasksFiles[index_refer_img]
        reference_img_file = listReferenceImgsFiles[index_refer_img]
        print("And: \'%s\'..." % (basename(in_refermask_file)))
        print("Reference image file: \'%s\'..." % (basename(reference_img_file)))


        size_procimage = FileReader.getImageSize(in_procdata_file)
        refermask_array = FileReader.getImageArray(in_refermask_file)
        size_fullimage = refermask_array.shape

        images_reconstructor.update_image_data(size_procimage, is_compute_normfact=False)

        num_samples_total = images_reconstructor.slidingWindow_generator.get_num_images()
        num_samples_3dirs = images_reconstructor.slidingWindow_generator.get_num_images_dirs()
        print("sliding-window gen. from image \'%s\' of size: \'%s\': num samples \'%s\', in local dirs: \'%s\'"
              %(ifile, str(args.size_in_images), num_samples_total, str(num_samples_3dirs)))
        limits_images_dirs = images_reconstructor.slidingWindow_generator.get_limits_sliding_window_image()
        for i in range(len(num_samples_3dirs)):
            print("coords of images in dir \'%s\': \'%s\'..." % (i, limits_images_dirs[i]))
        # endfor


        fill_slidewin_overlap_array = images_reconstructor.check_filling_overlap_image_samples()

        print("Proc image dims: %s..." % (str(fill_slidewin_overlap_array.shape)))

        # reconstruct from cropped / rescaled images
        if (args.cropImages):
            crop_bounding_box = dict_cropBoundingBoxes[filenamenoextension(reference_img_file)]
            print("Predicted data are cropped. Extend array size to original. Bounding-box: \'%s\'..." %(str(crop_bounding_box)))

            fill_slidewin_overlap_array = ExtendImages.compute3D(fill_slidewin_overlap_array, crop_bounding_box, size_fullimage)
            print("Final dims: %s..." %(str(fill_slidewin_overlap_array.shape)))


        outputfilename = nameOutputFiles(filenamenoextension(reference_img_file), args.size_in_images, args.slidewin_propOverlap)
        outputfilename = joinpathnames(OutputFilesPath, outputfilename)
        print("Output: \'%s\', of dims \'%s\'..." % (basename(outputfilename), fill_slidewin_overlap_array.shape))

        FileReader.writeImageArray(outputfilename, fill_slidewin_overlap_array)


        # compute the overlap with the refer. masks to know whether there are branches missing in the analysis
        masks_outside_fill_slidewin_overlap_array = np.where(fill_slidewin_overlap_array==0, refermask_array, 0)
        masks_outside_fill_unique_values = np.unique(masks_outside_fill_slidewin_overlap_array)
        if 1 in masks_outside_fill_unique_values:
            message = "Found masks outside filling overlap: the sliding-window does not cover some branches in ground-truth..."
            CatchWarningException(message)
            outputfilename = outputfilename.replace('test_', 'test_masks_outside_slideWin_')
            FileReader.writeImageArray(outputfilename, masks_outside_fill_slidewin_overlap_array)
    # endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--size_in_images', type=str2tupleint, default=IMAGES_DIMS_Z_X_Y)
    parser.add_argument('--isValidConvolutions', type=str2bool, default=ISVALIDCONVOLUTIONS)
    parser.add_argument('--filterPredictProbMaps', type=str2bool, default=FILTERPREDICTPROBMAPS)
    parser.add_argument('--prop_valid_outUnet', type=float, default=PROP_VALID_OUTUNET)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--slidewin_propOverlap', type=str2tuplefloat, default=SLIDEWINDOW_PROPOVERLAP_Z_X_Y)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))
    main(args)
