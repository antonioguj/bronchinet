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
from Common.WorkDirsManager import *
from DataLoaders.FileReaders import *
from Preprocessing.OperationImages import *
TYPE_ANIMATION = '1'
if (TYPE_ANIMATION == '1'):
    import imageio
elif (TYPE_ANIMATION == '2'):
    pass
    #import skimage
import argparse

WHITE_COLOR  = [255, 255, 255]
BLACK_COLOR  = [0, 0, 0]
RED_COLOR    = [255, 0, 0]
GREEN_COLOR  = [0, 255, 0]
BLUE_COLOR   = [0, 0, 255]
YELLOW_COLOR = [255, 255, 0]
MAGENTA_COLOR= [255, 0, 255]
CYAN_COLOR   = [0, 255, 255]



def main(args):
    # ---------- SETTINGS ----------
    nameInputPredictMasksRelPath= args.inputpredictiondir
    nameInputImagesRelPath      = 'Images_Proc/'
    nameInputReferMasksRelPath  = 'Airways_Proc/'
    nameInputPredictMasksFiles  = 'predict_binmasks*thres0-5_withtraquea.nii.gz'
    nameInputImagesFiles        = '*.nii.gz'
    nameInputReferMasksFiles    = '*outerwall*traquea.nii.gz'
    nameOutputRelPath           = 'Movies_Results/'
    temp_outfilename            = '%s_video.gif'
    prefixPatternInputFiles     = 'vol[0-9][0-9]_*'
    # ---------- SETTINGS ----------


    workDirsManager       = WorkDirsManager(args.datadir)
    InputPredictMasksPath = workDirsManager.getNameExistPath        (nameInputPredictMasksRelPath)
    InputImagesPath       = workDirsManager.getNameExistBaseDataPath(nameInputImagesRelPath)
    InputReferMasksPath   = workDirsManager.getNameExistBaseDataPath(nameInputReferMasksRelPath)
    OutputPath            = workDirsManager.getNameNewPath          (nameOutputRelPath)

    listInputPredictMasksFiles = findFilesDirAndCheck(InputPredictMasksPath,nameInputPredictMasksFiles)
    listInputImagesFiles       = findFilesDirAndCheck(InputImagesPath,      nameInputImagesFiles)
    listInputReferMasksFiles   = findFilesDirAndCheck(InputReferMasksPath,  nameInputReferMasksFiles)



    for i, in_predictmask_file in enumerate(listInputPredictMasksFiles):
        print("\nInput: \'%s\'..." % (in_predictmask_file))

        in_image_file = findFileWithSamePrefix(basename(in_predictmask_file), listInputImagesFiles,
                                               prefix_pattern=prefixPatternInputFiles)
        in_refermask_file = findFileWithSamePrefix(basename(in_predictmask_file), listInputReferMasksFiles,
                                                   prefix_pattern=prefixPatternInputFiles)
        #endfor
        print("assigned to \'%s\' and \'%s\'..." %(basename(in_image_file), basename(in_refermask_file)))

        in_image_array       = FileReader.getImageArray(in_image_file)
        in_predictmask_array = FileReader.getImageArray(in_predictmask_file)
        in_refermask_array   = FileReader.getImageArray(in_refermask_file)


        print("Rendering animations...")
        out_list_frames = []

        for i in range(in_image_array.shape[0]):
            in_image_slice       = in_image_array      [i,:,:]
            in_refermask_slice   = in_refermask_array  [i,:,:]
            in_predictmask_slice = in_predictmask_array[i,:,:]

            frame_image = (in_image_slice - np.min(in_image_slice)) / float(np.max(in_image_slice) - np.min(in_image_slice))
            frame_new = np.zeros((in_image_slice.shape[0], in_image_slice.shape[1], 3), dtype=np.uint8)
            frame_new[:, :, :] = 255 * frame_image[:, :, None]

            if (TYPE_ANIMATION == '1'):
                index_frame_TP_mask = np.argwhere(in_refermask_slice * in_predictmask_slice)
                index_frame_FN_mask = np.argwhere(in_refermask_slice * (1.0 - in_predictmask_slice))
                index_frame_FP_mask = np.argwhere((1.0 - in_refermask_slice) * in_predictmask_slice)

                # paint True Positives, False Negatives and False Positives in yellow, blue and red colour, respectively
                for index in index_frame_TP_mask:
                    frame_new[tuple(index)] = YELLOW_COLOR
                for index in index_frame_FN_mask:
                    frame_new[tuple(index)] = BLUE_COLOR
                for index in index_frame_FP_mask:
                    frame_new[tuple(index)] = RED_COLOR

                is_valid_frame = len(index_frame_TP_mask) > 0 or len(index_frame_FN_mask) > 0 or len(index_frame_FP_mask) > 0

            elif (TYPE_ANIMATION == '2'):
                index_frame_predict_bound_mask = skimage.segmentation.find_boundaries(in_predictmask_slice)
                index_frame_grndtruth_bound_mask = skimage.segmentation.find_boundaries(in_refermask_slice)

                # draw boundaries of prediction / ground-truth masks with green / red colour, respectively
                for index in index_frame_predict_bound_mask:
                    frame_new[tuple(index)] = GREEN_COLOR
                for index in index_frame_grndtruth_bound_mask:
                    frame_new[tuple(index)] = RED_COLOR

                is_valid_frame = len(index_frame_predict_bound_mask) > 0 or len(index_frame_grndtruth_bound_mask) > 0

            # skip frames that do not contain any predictions and/or ground-truth masks
            if is_valid_frame:
                out_list_frames.append(frame_new)

        if len(out_list_frames) > 0:
            print("Good movie...")
            prefix_casename = getSubstringPatternFilename(basename(in_predictmask_file),
                                                          substr_pattern=prefixPatternInputFiles)
            out_filename = joinpathnames(OutputPath, temp_outfilename %(prefix_casename))
            imageio.mimsave(out_filename, out_list_frames, fps=20)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('inputpredictiondir', type=str, default='Predictions_NEW')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)