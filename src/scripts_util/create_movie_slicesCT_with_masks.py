
import numpy as np
import argparse

from common.functionutil import makedir, join_path_names, list_files_dir, basename, get_substring_filename, \
    get_regex_pattern_filename, find_file_inlist_with_pattern
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.imageoperator import NormaliseImage

TYPE_ANIMATION = '1'
if TYPE_ANIMATION == '1':
    import imageio
elif TYPE_ANIMATION == '2':
    import skimage

WHITE_COLOR = [255, 255, 255]
BLACK_COLOR = [0, 0, 0]
RED_COLOR = [255, 0, 0]
GREEN_COLOR = [0, 255, 0]
BLUE_COLOR = [0, 0, 255]
YELLOW_COLOR = [255, 255, 0]
MAGENTA_COLOR = [255, 0, 255]
CYAN_COLOR = [0, 255, 255]


def main(args):

    list_input_images_files = list_files_dir(args.input_images_dir)
    list_input_predictions_files = list_files_dir(args.input_predictions_dir)
    list_input_reference_files = list_files_dir(args.inputrefermasksdir)
    pattern_search_infile = get_regex_pattern_filename(list_input_images_files[0])
    template_outvideo_filename = 'video_%s_preds.gif'

    makedir(args.output_dir)

    for i, in_prediction_file in enumerate(list_input_predictions_files):
        print("\nInput: \'%s\'..." % (in_prediction_file))

        in_image_file = find_file_inlist_with_pattern(basename(in_prediction_file), list_input_images_files,
                                                      pattern_search=pattern_search_infile)
        in_reference_file = find_file_inlist_with_pattern(basename(in_prediction_file), list_input_reference_files,
                                                          pattern_search=pattern_search_infile)
        print("Assigned to \'%s\' and \'%s\'..." % (basename(in_image_file), basename(in_reference_file)))

        in_image = ImageFileReader.get_image(in_image_file)
        in_prediction = ImageFileReader.get_image(in_prediction_file)
        in_reference = ImageFileReader.get_image(in_reference_file)

        print("Compute Rendered Animations...")
        out_list_frames = []

        for i in range(in_image.shape[0]):
            in_image_slice = in_image[i, :, :]
            in_prediction_slice = in_prediction[i, :, :]
            in_reference_slice = in_reference[i, :, :]

            frame_image = NormaliseImage.compute(in_image_slice)

            frame_new = np.zeros((in_image_slice.shape[0], in_image_slice.shape[1], 3), dtype=np.uint8)
            frame_new[:, :, :] = 255 * frame_image[:, :, None]

            if TYPE_ANIMATION == '1':
                indexes_frame_tp_mask = np.argwhere(in_prediction_slice * in_reference_slice)
                indexes_frame_fp_mask = np.argwhere(in_prediction_slice * (1 - in_reference_slice))
                indexes_frame_fn_mask = np.argwhere((1 - in_prediction_slice) * in_reference_slice)

                # paint True Positives, False Negatives and False Positives in yellow, blue and red colour, respectively
                for index in indexes_frame_tp_mask:
                    frame_new[tuple(index)] = YELLOW_COLOR
                for index in indexes_frame_fn_mask:
                    frame_new[tuple(index)] = BLUE_COLOR
                for index in indexes_frame_fp_mask:
                    frame_new[tuple(index)] = RED_COLOR

                # only accept frames that contain any (TP, FP, FN) mask
                is_valid_frame = \
                    len(indexes_frame_tp_mask) > 0 or len(indexes_frame_fn_mask) > 0 or len(indexes_frame_fp_mask) > 0

            elif TYPE_ANIMATION == '2':
                index_frame_prediction_bound_mask = skimage.segmentation.find_boundaries(in_prediction_slice)
                index_frame_reference_bound_mask = skimage.segmentation.find_boundaries(in_reference_slice)

                # draw boundaries of prediction / ground-truth masks with green / red colour, respectively
                for index in index_frame_prediction_bound_mask:
                    frame_new[tuple(index)] = GREEN_COLOR
                for index in index_frame_reference_bound_mask:
                    frame_new[tuple(index)] = RED_COLOR

                # only accept frames that contain any (prediction or reference) mask
                is_valid_frame = len(index_frame_prediction_bound_mask) > 0 or len(index_frame_reference_bound_mask) > 0

            else:
                is_valid_frame = None

            # skip frames that do not contain any masks
            if is_valid_frame:
                out_list_frames.append(frame_new)
        # endfor

        if len(out_list_frames) > 0:
            print("Create movie containing \'%s\' frames..." % (len(out_list_frames)))
            suffix_casename = get_substring_filename(basename(in_prediction_file), pattern_search=pattern_search_infile)
            out_filename = template_outvideo_filename % (suffix_casename)
            out_filename = join_path_names(args.output_dir, out_filename)
            imageio.mimsave(out_filename, out_list_frames, fps=20)
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_images_dir', type=str)
    parser.add_argument('input_predictions_dir', type=str)
    parser.add_argument('input_reference_dir', type=str)
    parser.add_argument('output_dir', type=str, default='./Movies_Result/')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
