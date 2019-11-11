#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from DataLoaders.BatchDataGenerator import *
from DataLoaders.FileReaders import *
from Preprocessing.ImageGeneratorManager import *
from Common.WorkDirsManager import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    nameInputImagesRelPath  = 'Images_WorkData/'
    nameInputLabelsRelPath  = 'Labels_WorkData/'
    nameVisualImagesRelPath = 'VisualWorkData/'
    nameImagesFiles         = '*.npz'
    nameLabelsFiles         = '*.npz'

    if (args.createImagesBatches):
        nameOutputVisualImagesFiles = 'visualImages-%0.2i_dim%s-batch%0.2i.nii.gz'
        nameOutputVisualLabelsFiles = 'visualLabels-%0.2i_dim%s-batch%0.2i.nii.gz'
    else:
        nameOutputVisualImagesFiles = 'visualImages-%0.2i_dim%s-batch%s.nii.gz'
        nameOutputVisualLabelsFiles = 'visualLabels-%0.2i_dim%s-batch%s.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager  = WorkDirsManager(args.datadir)
    InputImagesPath  = workDirsManager.getNameExistPath(nameInputImagesRelPath)
    InputLabelsPath  = workDirsManager.getNameExistPath(nameInputLabelsRelPath)
    VisualImagesPath = workDirsManager.getNameNewPath  (nameVisualImagesRelPath)

    listInputImagesFiles = findFilesDir(InputImagesPath, nameImagesFiles)
    listInputLabelsFiles = findFilesDir(InputLabelsPath, nameLabelsFiles)

    if (len(listInputImagesFiles) != len(listInputLabelsFiles)):
        message = 'num files in dir 1 \'%s\', not equal to num files in dir 2 \'%i\'...' %(len(listInputImagesFiles),
                                                                                           len(listInputLabelsFiles))
        CatchErrorException(message)



    for i, (in_image_file, in_label_file) in enumerate(zip(listInputImagesFiles, listInputLabelsFiles)):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))
        print("And: \'%s\'..." % (basename(in_label_file)))

        (in_image_array, in_label_array) = FileReader.get2ImageArraysAndCheck(in_image_file, in_label_file)
        print("Original dims : \'%s\'..." %(str(in_image_array.shape)))


        if (args.createImagesBatches):
            in_batches_shape = in_image_array.shape[1:]
            print("Input work data stored as batches of size  \'%s\'. Visualize batches..." % (str(in_batches_shape)))

            images_generator = getImagesVolumeTransformator3D(in_image_array,
                                                              args.transformationImages,
                                                              args.elasticDeformationImages)

            (out_visualimage_array, out_visuallabel_array) = images_generator.get_images_array(in_image_array, masks_array=in_label_array)

            for j, (in_batchimage_array, in_batchlabel_array) in enumerate(zip(out_visualimage_array, out_visuallabel_array)):

                out_image_filename = joinpathnames(VisualImagesPath, nameOutputVisualImagesFiles % (i+1, tuple2str(out_visualimage_array.shape[1:]), j+1))
                out_label_filename = joinpathnames(VisualImagesPath, nameOutputVisualLabelsFiles % (i+1, tuple2str(out_visuallabel_array.shape[1:]), j+1))

                FileReader.writeImageArray(out_image_filename, in_batchimage_array)
                FileReader.writeImageArray(out_label_filename, in_batchlabel_array)
            # endfor
        else:
            print("Input work data stored as volume. Generate batches of size \'%s\'. Visualize batches..." % (str(IMAGES_DIMS_Z_X_Y)))

            images_generator = getImagesDataGenerator3D(args.slidingWindowImages,
                                                        args.prop_overlap_Z_X_Y,
                                                        args.transformationImages,
                                                        args.elasticDeformationImages)
            batch_data_generator = BatchDataGenerator_2Arrays(IMAGES_DIMS_Z_X_Y,
                                                              in_image_array,
                                                              in_label_array,
                                                              images_generator,
                                                              size_batch=1,
                                                              shuffle=False)
            num_batches_total = len(batch_data_generator)
            print("Generate total \'%s\' batches by sliding-window, with coordinates:..." %(num_batches_total))

            for j in range(num_batches_total):
                coords_sliding_window_box = images_generator.slidingWindow_generator.get_limits_image(j)

                (out_visualimage_array, out_visuallabel_array) = next(batch_data_generator)

                out_visualimage_array = np.squeeze(out_visualimage_array, axis=0)
                out_visuallabel_array = np.squeeze(out_visuallabel_array, axis=0)

                out_image_filename = joinpathnames(VisualImagesPath, nameOutputVisualImagesFiles % (i+1, tuple2str(out_visualimage_array.shape), tuple2str(coords_sliding_window_box)))
                out_label_filename = joinpathnames(VisualImagesPath, nameOutputVisualLabelsFiles % (i+1, tuple2str(out_visuallabel_array.shape), tuple2str(coords_sliding_window_box)))

                FileReader.writeImageArray(out_image_filename, out_visualimage_array)
                FileReader.writeImageArray(out_label_filename, out_visuallabel_array)
            # endfor
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--slidewin_propOverlap', type=str2tuplefloat, default=SLIDEWIN_PROPOVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=ELASTICDEFORMATIONIMAGES)
    parser.add_argument('--createImagesBatches', type=str2bool, default=CREATEIMAGESBATCHES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
