
from common.constant import *
from common.functionutil import *
from common.workdirmanager import TrainDirManager
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.imageoperator import *
from imageoperators.maskoperator import MaskOperator
import argparse



def main(args):
    # ---------- SETTINGS ----------
    name_output_binary_masks_files = lambda in_name, thres: basename(in_name).replace('probmap','binmask_thres%s') %(str(thres).replace('.',''))
    # ---------- SETTINGS ----------


    workdir_manager             = TrainDirManager(args.basedir)
    input_posteriors_path       = workdir_manager.get_pathdir_exist(args.name_input_posteriors_relpath)
    output_binary_masks_path    = workdir_manager.get_pathdir_new(args.name_output_binary_masks_relpath)
    list_input_posteriors_files = list_files_dir(input_posteriors_path)
    prefix_pattern_input_files  = get_prefix_pattern_filename(list_input_posteriors_files[0])

    if (args.is_attach_coarse_airways):
        input_coarse_airways_path       = workdir_manager.get_datafile_exist(args.name_input_coarse_airways_relpath)
        list_input_coarse_airways_files = list_files_dir(input_coarse_airways_path)


    print("Compute \'%s\' Binary Masks from the Posteriors, using thresholding values: \'%s\'..." % (len(args.post_threshold_values),
                                                                                                     args.post_threshold_values))

    for i, in_posterior_file in enumerate(list_input_posteriors_files):
        print("\nInput: \'%s\'..." % (basename(in_posterior_file)))

        inout_posterior = ImageFileReader.get_image(in_posterior_file)
        print("Original dims : \'%s\'..." % (str(inout_posterior.shape)))

        in_metadata_file = ImageFileReader.get_image_metadata_info(in_posterior_file)


        if (args.is_attach_coarse_airways):
            print("Attach Trachea and Main Bronchi mask to complete the computed Binary Masks...")
            in_coarse_airways_file = find_file_inlist_same_prefix(basename(in_posterior_file), list_input_coarse_airways_files,
                                                                  prefix_pattern=prefix_pattern_input_files)
            print("Coarse Airways mask file: \'%s\'..." % (basename(in_coarse_airways_file)))

            in_coarse_airways = ImageFileReader.get_image(in_coarse_airways_file)


        for ithreshold in args.post_threshold_values:
            print("Compute Binary Masks thresholded to \'%s\'..." %(ithreshold))

            out_binary_mask = ThresholdImage.compute(inout_posterior, ithreshold)

            if (args.is_attach_coarse_airways):
                out_binary_mask = MaskOperator.merge_two_masks(out_binary_mask, in_coarse_airways) #isNot_intersect_masks=True)


            # Output predicted binary masks
            output_binary_mask_file = join_path_names(output_binary_masks_path, name_output_binary_masks_files(in_posterior_file, ithreshold))
            print("Output: \'%s\', of dims \'%s\'..." % (basename(output_binary_mask_file), str(out_binary_mask.shape)))

            ImageFileReader.write_image(output_binary_mask_file, out_binary_mask, metadata=in_metadata_file)
        # endfor
    # endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--name_input_posteriors_relpath', type=str, default=NAME_POSTERIORS_RELPATH)
    parser.add_argument('--name_input_coarse_airways_relpath', type=str, default=NAME_RAW_COARSEAIRWAYS_RELPATH)
    parser.add_argument('--name_output_binary_masks_relpath', type=str, default=NAME_PRED_BINARYMASKS_RELPATH)
    parser.add_argument('--post_threshold_values', type=float, default=POST_THRESHOLD_VALUE)
    parser.add_argument('--is_attach_coarse_airways', type=str2bool, default=IS_ATTACH_COARSE_AIRWAYS)
    args = parser.parse_args()

    if type(args.post_threshold_values) in [int, float]:
        args.post_threshold_values = [args.post_threshold_values]

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)