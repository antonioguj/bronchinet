
import argparse

from common.constant import BASEDIR, POST_THRESHOLD_VALUE, IS_ATTACH_COARSE_AIRWAYS, \
                            NAME_POSTERIORS_RELPATH, NAME_PRED_BINARYMASKS_RELPATH, \
                            NAME_RAW_COARSEAIRWAYS_RELPATH, NAME_REFERENCE_KEYS_PROCIMAGE_FILE
from common.functionutil import join_path_names, basename, list_files_dir, get_pattern_refer_filename, \
                                find_file_inlist_same_prefix, str2bool, read_dictionary
from common.workdirmanager import TrainDirManager
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.imageoperator import ThresholdImage
from imageoperators.maskoperator import MaskOperator


def main(args):

    # SETTINGS
    def name_output_binary_masks_files(in_name: str, thres: float):
        return basename(in_name).replace('probmap', 'binmask_thres%s') % (str(thres).replace('.', ''))
    # --------

    workdir_manager = TrainDirManager(args.basedir)
    input_posteriors_path = workdir_manager.get_pathdir_exist(args.name_input_posteriors_relpath)
    in_reference_keys_file = workdir_manager.get_datafile_exist(args.name_input_reference_keys_file)
    output_binary_masks_path = workdir_manager.get_pathdir_new(args.name_output_binary_masks_relpath)
    list_input_posteriors_files = list_files_dir(input_posteriors_path)
    indict_reference_keys = read_dictionary(in_reference_keys_file)
    pattern_search_input_files = get_pattern_refer_filename(list(indict_reference_keys.values())[0])

    if (args.is_attach_coarse_airways):
        input_coarse_airways_path = workdir_manager.get_datadir_exist(args.name_input_coarse_airways_relpath)
        list_input_coarse_airways_files = list_files_dir(input_coarse_airways_path)

    # *****************************************************

    print("Compute \'%s\' Binary Masks from the Posteriors, using thresholding values: \'%s\'..."
          % (len(args.post_threshold_values), args.post_threshold_values))

    for i, in_posterior_file in enumerate(list_input_posteriors_files):
        print("\nInput: \'%s\'..." % (basename(in_posterior_file)))

        inout_posterior = ImageFileReader.get_image(in_posterior_file)
        print("Original dims : \'%s\'..." % (str(inout_posterior.shape)))

        in_metadata_file = ImageFileReader.get_image_metadata_info(in_posterior_file)

        if (args.is_attach_coarse_airways):
            print("Attach Trachea and Main Bronchi mask to complete the computed Binary Masks...")
            in_coarse_airways_file = find_file_inlist_same_prefix(basename(in_posterior_file),
                                                                  list_input_coarse_airways_files,
                                                                  pattern_prefix=pattern_search_input_files)
            print("Coarse Airways mask file: \'%s\'..." % (basename(in_coarse_airways_file)))

            in_coarse_airways = ImageFileReader.get_image(in_coarse_airways_file)

        for ithreshold in args.post_threshold_values:
            print("Compute Binary Masks thresholded to \'%s\'..." % (ithreshold))

            out_binary_mask = ThresholdImage.compute(inout_posterior, ithreshold)

            if (args.is_attach_coarse_airways):
                out_binary_mask = MaskOperator.merge_two_masks(out_binary_mask, in_coarse_airways)
                # isNot_intersect_masks=True)

            # Output predicted binary masks
            output_binary_mask_file = join_path_names(output_binary_masks_path,
                                                      name_output_binary_masks_files(in_posterior_file, ithreshold))
            print("Output: \'%s\', of dims \'%s\'..." % (basename(output_binary_mask_file), str(out_binary_mask.shape)))

            ImageFileReader.write_image(output_binary_mask_file, out_binary_mask, metadata=in_metadata_file)
        # endfor
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--post_threshold_values', type=float, default=POST_THRESHOLD_VALUE)
    parser.add_argument('--is_attach_coarse_airways', type=str2bool, default=IS_ATTACH_COARSE_AIRWAYS)
    parser.add_argument('--name_input_posteriors_relpath', type=str, default=NAME_POSTERIORS_RELPATH)
    parser.add_argument('--name_input_coarse_airways_relpath', type=str, default=NAME_RAW_COARSEAIRWAYS_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--name_output_binary_masks_relpath', type=str, default=NAME_PRED_BINARYMASKS_RELPATH)
    args = parser.parse_args()

    if type(args.post_threshold_values) in [int, float]:
        args.post_threshold_values = [args.post_threshold_values]

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
