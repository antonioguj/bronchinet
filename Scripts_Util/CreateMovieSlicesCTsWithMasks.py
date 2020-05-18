#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.FunctionsUtil import *
from DataLoaders.FileReaders import *
from OperationImages.OperationImages import *
import argparse

TYPE_ANIMATION = '1'
if (TYPE_ANIMATION == '1'):
    import imageio
elif (TYPE_ANIMATION == '2'):
    import skimage

WHITE_COLOR  = [255, 255, 255]
BLACK_COLOR  = [0, 0, 0]
RED_COLOR    = [255, 0, 0]
GREEN_COLOR  = [0, 255, 0]
BLUE_COLOR   = [0, 0, 255]
YELLOW_COLOR = [255, 255, 0]
MAGENTA_COLOR= [255, 0, 255]
CYAN_COLOR   = [0, 255, 255]



def main(args):

    list_input_images_files      = findFilesDirAndCheck(args.inputimagesdir)
    list_input_predictions_files = findFilesDirAndCheck(args.inputpredictionsdir)
    list_input_reference_files   = findFilesDirAndCheck(args.inputrefermasksdir)
    prefix_pattern_input_files   = getFilePrefixPattern(list_input_predictions_files[0])

    template_outvideo_filename   = 'video_%s_preds.gif'

    makedir(args.outputdir)



    for i, in_prediction_file in enumerate(list_input_predictions_files):
        print("\nInput: \'%s\'..." % (in_prediction_file))

        in_image_file = findFileWithSamePrefixPattern(basename(in_prediction_file), list_input_images_files,
                                                      prefix_pattern=prefix_pattern_input_files)
        in_reference_file = findFileWithSamePrefixPattern(basename(in_prediction_file), list_input_reference_files,
                                                          prefix_pattern=prefix_pattern_input_files)
        #endfor
        print("Assigned to \'%s\' and \'%s\'..." %(basename(in_image_file), basename(in_reference_file)))

        in_image_array      = FileReader.get_image_array(in_image_file)
        in_prediction_array = FileReader.get_image_array(in_prediction_file)
        in_reference_array  = FileReader.get_image_array(in_reference_file)


        print("Compute Rendered Animations...")
        out_list_frames = []

        for i in range(in_image_array.shape[0]):
            in_image_slice      = in_image_array     [i,:,:]
            in_prediction_slice = in_prediction_array[i,:,:]
            in_reference_slice  = in_reference_array [i,:,:]

            frame_image = NormaliseImages.compute3D(in_image_slice)

            frame_new = np.zeros((in_image_slice.shape[0], in_image_slice.shape[1], 3), dtype=np.uint8)
            frame_new[:, :, :] = 255 * frame_image[:, :, None]

            if (TYPE_ANIMATION == '1'):
                index_frame_TP_mask = np.argwhere(in_prediction_slice * in_reference_slice)
                index_frame_FP_mask = np.argwhere(in_prediction_slice * (1 - in_reference_slice))
                index_frame_FN_mask = np.argwhere((1 - in_prediction_slice) * in_reference_slice)

                # paint True Positives, False Negatives and False Positives in yellow, blue and red colour, respectively
                for index in index_frame_TP_mask:
                    frame_new[tuple(index)] = YELLOW_COLOR
                for index in index_frame_FN_mask:
                    frame_new[tuple(index)] = BLUE_COLOR
                for index in index_frame_FP_mask:
                    frame_new[tuple(index)] = RED_COLOR

                # only accept frames that contain any (TP, FP, FN) mask
                is_valid_frame = len(index_frame_TP_mask) > 0 or len(index_frame_FN_mask) > 0 or len(index_frame_FP_mask) > 0

            elif (TYPE_ANIMATION == '2'):
                index_frame_prediction_bound_mask = skimage.segmentation.find_boundaries(in_prediction_slice)
                index_frame_reference_bound_mask  = skimage.segmentation.find_boundaries(in_reference_slice)

                # draw boundaries of prediction / ground-truth masks with green / red colour, respectively
                for index in index_frame_prediction_bound_mask:
                    frame_new[tuple(index)] = GREEN_COLOR
                for index in index_frame_reference_bound_mask:
                    frame_new[tuple(index)] = RED_COLOR

                # only accept frames that contain any (prediction or reference) mask
                is_valid_frame = len(index_frame_prediction_bound_mask) > 0 or len(index_frame_reference_bound_mask) > 0

            # skip frames that do not contain any masks
            if is_valid_frame:
                out_list_frames.append(frame_new)
        # endfor

        if len(out_list_frames) > 0:
            print("Create movie containing \'%s\' frames..." % (len(out_list_frames)))
            suffix_casename = getSubstringPatternFilename(basename(in_prediction_file),
                                                          substr_pattern=prefix_pattern_input_files)

            out_filename = template_outvideo_filename %(suffix_casename)
            out_filename = joinpathnames(args.outputdir, out_filename)
            imageio.mimsave(out_filename, out_list_frames, fps=20)
    # endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimagesdir', type=str)
    parser.add_argument('inputpredictionsdir', type=str)
    parser.add_argument('inputreferencedir', type=str)
    parser.add_argument('outputdir', type=str, default='./Movies_Result/')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)