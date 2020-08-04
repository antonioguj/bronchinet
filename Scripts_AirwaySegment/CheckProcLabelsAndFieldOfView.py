#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from common.workdir_manager import *
from dataloaders.batchdatagenerator_manager import *
from dataloaders.loadimagedata_manager import *
from networks.pytorch.networks import *
from imageoperators.imageoperator import *
from postprocessing.imagereconstructor_manager import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    nameInputImagesFiles = 'images_proc*.nii.gz'
    nameInputLabelsFiles = 'labels_proc*.nii.gz'

    def nameOutputFiles(in_name, in_size_image):
        suffix = 'outsize-%s' %('x'.join([str(s) for s in in_size_image]))
        return 'fieldOfView_' + basename_file_noext(in_name) + '_' + suffix + '.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager    = GeneralDirManager(args.datadir)
    InputImagesDataPath= workDirsManager.get_pathdir_exist(args.nameInputImagesRelPath)
    InputLabelsDataPath= workDirsManager.get_pathdir_exist(args.nameInputLabelsRelPath)
    InputReferKeysFile = workDirsManager.get_pathfile_exist(args.nameInputReferKeysFile)
    OutputFilesPath    = workDirsManager.get_pathdir_new(args.outputdir)

    listInputLabelsFiles = list_files_dir(InputLabelsDataPath, nameInputLabelsFiles)


    # Build model to calculate the output size
    model_net = DICTAVAILMODELS3D(args._size_image_in,
                                  num_levels=args.num_layers,
                                  num_featmaps_in=1,
                                  isUse_valid_convols=args.isValidConvolutions)

    size_output_modelnet = tuple(model_net.get_size_output()[1:])
    if args.isValidConvolutions:
        print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..." % (str(args._size_image_in),
                                                                                               str(size_output_modelnet)))

    images_generator = get_images_generator(args._size_image_in,
                                            args.slidingWindowImages,
                                            args.propOverlapSlidingWindow,
                                            args.randomCropWindowImages,
                                            args.numRandomImagesPerVolumeEpoch)
    images_reconstructor = get_images_reconstructor(args._size_image_in,
                                                    args.slidingWindowImages,
                                                    args.propOverlapSlidingWindow,
                                                    args.randomCropWindowImages,
                                                    args.numRandomImagesPerVolumeEpoch,
                                                    is_output_nnet_validconvs=args.isValidConvolutions,
                                                    size_output_image=size_output_modelnet,
                                                    is_filter_output_nnet=FILTERPREDICTPROBMAPS,
                                                    prop_filter_output_nnet=PROP_VALID_OUTUNET)


    names_files_different = []

    for ifile, in_label_file in enumerate(listInputLabelsFiles):
        print("\nInput: \'%s\'..." % (basename(in_label_file)))

        in_label_array = ImageFileReader.get_image(in_label_file)
        print("Original dims : \'%s\'..." % (str(in_label_array.shape)))


        print("Loading data in batches...")
        if (args.slidingWindowImages or args.randomCropWindowImages or args.transformationRigidImages):
            in_label_data = LoadImageDataManager.load_1file(in_label_file)
            in_label_data = np.expand_dims(in_label_data, axis=-1)
            in_label_generator = get_batchdata_generator_with_generator(args._size_image_in,
                                                                        [in_label_data],
                                                                        [in_label_data],
                                                                        images_generator,
                                                                        batch_size=1,
                                                                        is_output_nnet_validconvs=args.isValidConvolutions,
                                                                        size_output_images=size_output_modelnet,
                                                                        shuffle=False,
                                                                        is_datagen_in_gpu=False)
            (_, inout_batches_label_arrays) = in_label_generator.get_full_data()
        else:
            inout_batches_label_arrays = LoadImageDataInBatchesManager(args._size_image_in).load_1file(in_label_file)
            inout_batches_label_arrays = np.expand_dims(inout_batches_label_arrays, axis=0)

        print("Total data batches generated: %s..." % (len(inout_batches_label_arrays)))


        print("Reconstruct data batches to full size...")
        # init reconstructor with size of "ifile"
        out_recons_image_shape = ImageFileReader.get_image_size(in_label_file)
        images_reconstructor.update_image_data(out_recons_image_shape)

        out_recons_label_array = images_reconstructor.compute(inout_batches_label_arrays)
        out_recons_fieldOfView_array = images_reconstructor.compute_overlap_image_patches()


        print("Compare voxels-wise Labels from both original and reconstructed arrays...")
        in_orig_onlylabels_array = np.where(in_label_array > 0, in_label_array, 0)
        in_recons_onlylabels_array = np.where(out_recons_label_array > 0, out_recons_label_array, 0)

        is_arrays_equal_voxelwise = np.array_equal(in_orig_onlylabels_array, in_recons_onlylabels_array)

        if is_arrays_equal_voxelwise:
            print("GOOD: Images are equal voxelwise...")
        else:
            print("WARNING: Images 1 and 2 are not equal...")
            num_voxels_labels_orig_array = np.sum(in_orig_onlylabels_array)
            num_voxels_labels_recons_array = np.sum(in_recons_onlylabels_array)
            print("Num Voxels for Labels in original and reconstructed arrays: \'%s\' != \'%s\'..." %(num_voxels_labels_orig_array,
                                                                                                      num_voxels_labels_recons_array))
            names_files_different.append(basename(in_label_file))


        out_filename = join_path_names(OutputFilesPath, nameOutputFiles(in_label_file, size_output_modelnet))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_filename), out_recons_fieldOfView_array.shape))

        ImageFileReader.write_image(out_filename, out_recons_fieldOfView_array)
    # endfor

    if (len(names_files_different) == 0):
        print("\nGOOD: ALL IMAGE FILES ARE EQUAL...")
    else:
        print("\nWARNING: Found \'%s\' files that are different. Names of files: \'%s\'..." %(len(names_files_different),
                                                                                              names_files_different))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('outputdir', type=str, default='CheckFieldOfViewNetworks/')
    parser.add_argument('--size_in_images', type=str2tuple_int, default=IMAGES_DIMS_Z_X_Y)
    parser.add_argument('--nameInputImagesRelPath', type=str, default=NAME_PROCIMAGES_RELPATH)
    parser.add_argument('--nameInputLabelsRelPath', type=str, default=NAME_PROCLABELS_RELPATH)
    parser.add_argument('--nameInputReferKeysFile', type=str, default=NAME_REFERKEYSPROCIMAGE_FILE)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--isValidConvolutions', type=str2bool, default=ISVALIDCONVOLUTIONS)
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=True)
    parser.add_argument('--propOverlapSlidingWindow', type=str2tuple_float, default=PROPOVERLAPSLIDINGWINDOW_TESTING_Z_X_Y)
    parser.add_argument('--randomCropWindowImages', type=str2tuple_float, default=False)
    parser.add_argument('--numRandomImagesPerVolumeEpoch', type=str2tuple_float, default=0)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))
    main(args)
