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
from Preprocessing.OperationsImages import *

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
    nameInputImagesRelPath = 'ProcImages'
    nameInputMasksRelPath  = 'ProcMasks'

    namePredictMasksFiles  = 'predict_binmasks*thres0-5_withtraquea.nii.gz'
    nameInputImagesFiles   = '*.nii.gz'
    nameInputMasksFiles    = '*outerwall*traquea.nii.gz'

    # template search files
    tempSearchInputFiles  = 'av[0-9]*'

    # create file to save FROC values
    temp_outfilename = '%s_video.gif'
    # ---------- SETTINGS ----------


    workDirsManager      = WorkDirsManager(args.basedir)
    BaseDataPath         = workDirsManager.getNameBaseDataPath()
    InputPredictMasksPath= workDirsManager.getNameExistPath(args.basedir, args.predictionsdir)
    InputImagesPath      = workDirsManager.getNameExistPath(args.basedir, nameInputImagesRelPath)
    InputMasksPath       = workDirsManager.getNameExistPath(args.basedir, nameInputMasksRelPath)
    OutputPath           = workDirsManager.getNameNewPath  (args.basedir, 'movies_results')

    listPredictMasksFiles   = findFilesDir(InputPredictMasksPath,namePredictMasksFiles)
    listImagesCTFiles       = findFilesDir(InputImagesPath,      nameInputImagesFiles)
    listGrndTruthMasksFiles = findFilesDir(InputMasksPath,       nameInputMasksFiles)

    nbPredictionsFiles    = len(listPredictMasksFiles)
    nbImagesCTFiles       = len(listImagesCTFiles)
    nbGrndTruthMasksFiles = len(listGrndTruthMasksFiles)

    # Run checkers
    if (nbPredictionsFiles == 0):
        message = "0 Predictions found in dir \'%s\'" %(InputPredictMasksPath)
        CatchErrorException(message)
    if (nbImagesCTFiles == 0):
        message = "0 Images CT found in dir \'%s\'" %(InputImagesPath)
        CatchErrorException(message)
    if (nbGrndTruthMasksFiles == 0):
        message = "0 Ground-truth Masks found in dir \'%s\'" %(InputMasksPath)
        CatchErrorException(message)



    for i, predict_masks_file in enumerate(listPredictMasksFiles):

        print('\'%s\'...' %(predict_masks_file))

        name_prefix_case = getExtractSubstringPattern(basename(predict_masks_file),
                                                      tempSearchInputFiles)

        for iterfile_1, iterfile_2 in zip(listImagesCTFiles,
                                          listGrndTruthMasksFiles):
            if name_prefix_case in iterfile_1:
                images_CT_file       = iterfile_1
                grndtruth_masks_file = iterfile_2
        #endfor
        print("assigned to '%s' and '%s'..." %(basename(images_CT_file),
                                               basename(grndtruth_masks_file)))

        predict_masks_array  = FileReader.getImageArray(predict_masks_file)
        images_CT_array      = FileReader.getImageArray(images_CT_file)
        grndtruth_masks_array= FileReader.getImageArray(grndtruth_masks_file)

        if (args.invertImageAxial):
            predict_masks_array  = FlippingImages.compute(predict_masks_array,   axis=0)
            images_CT_array      = FlippingImages.compute(images_CT_array,       axis=0)
            grndtruth_masks_array= FlippingImages.compute(grndtruth_masks_array, axis=0)



        print("Rendering animations...")
        list_frames = []

        for i in range(images_CT_array.shape[0]):

            images_CT_slice       = images_CT_array      [i,:,:]
            grndtruth_masks_slice = grndtruth_masks_array[i,:,:]
            predict_masks_slice   = predict_masks_array  [i,:,:]


            frame_image_CT = (images_CT_slice - np.min(images_CT_slice)) / float(np.max(images_CT_slice) - np.min(images_CT_slice))

            frame_new = np.zeros((images_CT_slice.shape[0], images_CT_slice.shape[1], 3), dtype=np.uint8)

            frame_new[:, :, :] = 255 * frame_image_CT[:, :, None]

            if (TYPE_ANIMATION == '1'):

                index_frame_TP_mask = np.argwhere(grndtruth_masks_slice * predict_masks_slice)
                index_frame_FN_mask = np.argwhere(grndtruth_masks_slice * (1.0 - predict_masks_slice))
                index_frame_FP_mask = np.argwhere((1.0 - grndtruth_masks_slice) * predict_masks_slice)

                # paint True Positives, False Negatives and False Positives in yellow, blue and red colour, respectively
                for index in index_frame_TP_mask:
                    frame_new[tuple(index)] = YELLOW_COLOR
                for index in index_frame_FN_mask:
                    frame_new[tuple(index)] = BLUE_COLOR
                for index in index_frame_FP_mask:
                    frame_new[tuple(index)] = RED_COLOR

                is_valid_frame = len(index_frame_TP_mask) > 0 or len(index_frame_FN_mask) > 0 or len(index_frame_FP_mask) > 0

            elif (TYPE_ANIMATION == '2'):

                index_frame_predict_bound_mask = skimage.segmentation.find_boundaries(predict_masks_slice)
                index_frame_grndtruth_bound_mask = skimage.segmentation.find_boundaries(grndtruth_masks_slice)

                # draw boundaries of prediction / ground-truth masks with green / red colour, respectively
                for index in index_frame_predict_bound_mask:
                    frame_new[tuple(index)] = GREEN_COLOR
                for index in index_frame_grndtruth_bound_mask:
                    frame_new[tuple(index)] = RED_COLOR

                is_valid_frame = len(index_frame_predict_bound_mask) > 0 or len(index_frame_grndtruth_bound_mask) > 0


            # skip frames that do not contain any predictions and/or ground-truth masks
            if is_valid_frame:
                list_frames.append(frame_new)

        if len(list_frames) > 0:
            print("Good movie...")
            outfilename = joinpathnames(OutputPath, temp_outfilename %(name_prefix_case))
            imageio.mimsave(outfilename, list_frames, fps=20)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--invertImageAxial', type=str2bool, default=INVERTIMAGEAXIAL)
    parser.add_argument('--predictionsdir', default='Predictions_NEW')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)