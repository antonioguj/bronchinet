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
    nameInputImagesRelPath  = 'ProcImagesExperData'
    nameInputMasksRelPath   = 'ProcMasksExperData'
    nameVisualImagesRelPath = 'VisualInputData'

    # Get the file list:
    nameInImagesFiles        = '*.npy'
    nameInMasksFiles         = '*.npy'
    nameOutImagesFiles_type1 = 'visualImages-%0.2i_dim%s.nii.gz'
    nameOutMasksFiles_type1  = 'visualMasks-%0.2i_dim%s.nii.gz'
    nameOutImagesFiles_type2 = 'visualImages-%0.2i_dim%s-batch%0.2i.nii.gz'
    nameOutMasksFiles_type2  = 'visualMasks-%0.2i_dim%s-batch%0.2i.nii.gz'
    nameOutImagesFiles_type3 = 'visualImages-%0.2i_dim%s-batch%s.nii.gz'
    nameOutMasksFiles_type3  = 'visualMasks-%0.2i_dim%s-batch%s.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameBaseDataPath()
    InputImagesPath  = workDirsManager.getNameExistPath(BaseDataPath, nameInputImagesRelPath)
    InputMasksPath   = workDirsManager.getNameExistPath(BaseDataPath, nameInputMasksRelPath )
    VisualImagesPath = workDirsManager.getNameNewPath  (args.basedir, nameVisualImagesRelPath)

    listImagesFiles = findFilesDir(InputImagesPath, nameInImagesFiles)
    listMasksFiles  = findFilesDir(InputMasksPath,  nameInMasksFiles)

    nbImagesFiles = len(listImagesFiles)
    nbMasksFiles  = len(listMasksFiles)

    # Run checkers
    if (nbImagesFiles == 0):
        message = "0 Images found in dir \'%s\'" %(InputImagesPath)
        CatchErrorException(message)
    if (nbImagesFiles != nbMasksFiles):
        message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
        CatchErrorException(message)



    for i, (images_file, masks_file) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' % (images_file))
        print('\'%s\'...' % (masks_file))

        images_array = FileReader.getImageArray(images_file)
        masks_array  = FileReader.getImageArray(masks_file)

        if (images_array.shape != masks_array.shape):
            message = "size of Images and Masks not equal: %s != %s" % (images_array.shape, masks_array.shape)
            CatchErrorException(message)
        print("Original image of size: %s..." %(str(images_array.shape)))


        if (args.createImagesBatches):
            shape_batches = images_array.shape[1:]
            print("Input images data stored as batches of size %s. Visualize batches..." % (str(shape_batches)))

            images_generator = getImagesVolumeTransformator3D(images_array,
                                                              args.transformationImages,
                                                              args.elasticDeformationImages)

            (visual_images_array, visual_masks_array) = images_generator.get_images_array(images_array, masks_array=masks_array)

            for j, (batch_images_array, batch_masks_array) in enumerate(zip(visual_images_array, visual_masks_array)):

                out_images_filename = joinpathnames(VisualImagesPath, nameOutImagesFiles_type2 % (i+1, tuple2str(visual_images_array.shape[1:]), j+1))
                out_masks_filename  = joinpathnames(VisualImagesPath, nameOutMasksFiles_type2  % (i+1, tuple2str(visual_masks_array .shape[1:]), j+1))

                FileReader.writeImageArray(out_images_filename, batch_images_array)
                FileReader.writeImageArray(out_masks_filename,  batch_masks_array )
            # endfor
        else:
            if (args.visualProcDataInBatches):
                print("Input images data stored as volume. Generate batches of size %s. Visualize batches..." % (str(IMAGES_DIMS_Z_X_Y)))

                images_generator = getImagesDataGenerator3D(args.slidingWindowImages,
                                                            args.prop_overlap_Z_X_Y,
                                                            args.transformationImages,
                                                            args.elasticDeformationImages)

                batch_data_generator = BatchDataGenerator_2Arrays(IMAGES_DIMS_Z_X_Y,
                                                                  images_array,
                                                                  masks_array,
                                                                  images_generator,
                                                                  size_batch=1,
                                                                  shuffle=False)
                num_batches_total = len(batch_data_generator)
                print("Generate total %s batches by sliding-window, with coordinates:..." %(num_batches_total))

                for j in range(num_batches_total):
                    coords_sliding_window_box = images_generator.slidingWindow_generator.get_limits_image(j)

                    (visual_images_array, visual_masks_array) = next(batch_data_generator)

                    visual_images_array = np.squeeze(visual_images_array, axis=0)
                    visual_masks_array  = np.squeeze(visual_masks_array,  axis=0)

                    out_images_filename = joinpathnames(VisualImagesPath, nameOutImagesFiles_type3 % (i+1, tuple2str(visual_images_array.shape), tuple2str(coords_sliding_window_box)))
                    out_masks_filename  = joinpathnames(VisualImagesPath, nameOutMasksFiles_type3  % (i+1, tuple2str(visual_masks_array.shape ), tuple2str(coords_sliding_window_box)))

                    FileReader.writeImageArray(out_images_filename, visual_images_array)
                    FileReader.writeImageArray(out_masks_filename,  visual_masks_array)
                # endfor
            else:
                print("Input images data stored as volume of size %s. Visualize volume..." % (str(images_array.shape)))

                images_generator = getImagesVolumeTransformator3D(images_array.shape,
                                                                  args.transformationImages,
                                                                  args.elasticDeformationImages)

                (visual_images_array, visual_masks_array) = images_generator.get_images_array(images_array, masks_array=masks_array)

                out_images_filename = joinpathnames(VisualImagesPath, nameOutImagesFiles_type1 % (i+1, tuple2str(visual_images_array.shape)))
                out_masks_filename  = joinpathnames(VisualImagesPath, nameOutMasksFiles_type1  % (i+1, tuple2str(visual_masks_array .shape)))

                FileReader.writeImageArray(out_images_filename, visual_images_array)
                FileReader.writeImageArray(out_masks_filename,  visual_masks_array)
    #endfor



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=ELASTICDEFORMATIONIMAGES)
    parser.add_argument('--createImagesBatches', type=str2bool, default=CREATEIMAGESBATCHES)
    parser.add_argument('--visualProcDataInBatches', type=str2bool, default=VISUALPROCDATAINBATCHES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
