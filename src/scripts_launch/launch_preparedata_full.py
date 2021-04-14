
import subprocess
import traceback
import sys
import argparse

from common.constant import SIZE_IN_IMAGES, IS_MASK_REGION_INTEREST, IS_CROP_IMAGES, IS_RESCALE_IMAGES, \
    NAME_RAW_IMAGES_RELPATH, NAME_RAW_LABELS_RELPATH, NAME_RAW_ROIMASKS_RELPATH, NAME_REFERENCE_FILES_RELPATH, \
    NAME_RAW_CENTRELINES_RELPATH, NAME_RAW_COARSEAIRWAYS_RELPATH, NAME_CROP_BOUNDBOXES_FILE, NAME_RESCALE_FACTORS_FILE,\
    IS_TWO_BOUNDBOXES_LUNGS
from common.functionutil import currentdir, makedir, set_dirname_suffix, join_path_names, list_files_dir, \
    basename, fileextension, str2bool, str2tuple_int
from common.exceptionmanager import catch_error_exception

CODEDIR = '/home/antonio/Codes/Antonio_repository/bronchinet/src/'
SCRIPT_CONVERT_TO_NIFTI = join_path_names(CODEDIR, 'scripts_util/convert_images_to_nifti.py')
SCRIPT_BINARISE_MASKS = join_path_names(CODEDIR, 'scripts_util/apply_operation_images.py')
SCRIPT_GET_TRACHEA_MAIN_BRONCHI = join_path_names(CODEDIR, 'scripts_util/apply_operation_images.py')
SCRIPT_COMPUTE_CENTRELINES = join_path_names(CODEDIR, 'scripts_util/apply_operation_images.py')
SCRIPT_RESCALE_ROI_MASKS = join_path_names(CODEDIR, 'scripts_util/apply_operation_images.py')
SCRIPT_EXTEND_CROPPED_IMAGES = \
    join_path_names(CODEDIR, 'scripts_preparedata/prepdata_dlcst/extend_cropped_images_fullsize.py')
SCRIPT_CALC_RESCALE_FACTOR_IMAGES = join_path_names(CODEDIR, 'scripts_preparedata/compute_rescalefactor_images.py')
SCRIPT_CALC_BOUNDING_BOX_IMAGES = join_path_names(CODEDIR, 'scripts_preparedata/compute_boundingbox_images.py')
SCRIPT_PREPARE_DATA = join_path_names(CODEDIR, 'scripts_preparedata/prepare_data.py')

SOURCE_REMOTE_BASEDIR = 'agarcia@bigr-app001:/scratch/agarcia/Data/'

LIST_TYPE_DATA_AVAIL = ['training', 'testing']


def print_call(new_call):
    message = ' '.join(new_call)
    print("\n" + "*" * 100)
    print("<<< Launch: %s >>>" % (message))
    print("*" * 100 + "\n")


def launch_call(new_call):
    popen_obj = subprocess.Popen(new_call)
    popen_obj.wait()


def create_task_replace_dirs(input_dir: str, input_dir_to_replace: str):
    new_call_1 = ['rm', '-r', input_dir]
    new_call_2 = ['mv', input_dir_to_replace, input_dir]
    return [new_call_1, new_call_2]


def create_task_decompress_data(input_data_dir: str, is_keep_files: bool):
    list_files = list_files_dir(input_data_dir)
    extension_file = fileextension(list_files[0])
    sublist_calls = []

    if extension_file == '.dcm.gz':
        # decompress data
        new_call = ['gunzip', '-vr', input_data_dir]
        sublist_calls.append(new_call)

    if is_keep_files and extension_file in ['.dcm', '.dcm.gz']:
        # convert to nifty, if we keep the raw images for testing
        new_input_data_dir = set_dirname_suffix(input_data_dir, 'Nifty')
        new_call = ['python3', SCRIPT_CONVERT_TO_NIFTI, input_data_dir, new_input_data_dir]
        sublist_calls.append(new_call)

        new_sublist_calls = create_task_replace_dirs(input_data_dir, new_input_data_dir)
        sublist_calls += new_sublist_calls

    return sublist_calls


def main(args):

    # output_datadir = update_dirname(args.output_datadir)
    output_datadir = args.output_datadir
    makedir(output_datadir)

    source_remote_datadir = join_path_names(SOURCE_REMOTE_BASEDIR, args.source_remote_datadir)

    name_remote_raw_images_path = join_path_names(source_remote_datadir, 'CTs/')
    name_remote_raw_labels_path = join_path_names(source_remote_datadir, 'Airways/')
    name_remote_raw_roimasks_path = join_path_names(source_remote_datadir, 'Lungs/')

    if args.is_prepare_coarse_airways:
        name_remote_raw_coarse_airways_path = join_path_names(source_remote_datadir, 'CoarseAirways/')
    else:
        name_remote_raw_coarse_airways_path = None

    if args.source_remote_datadir in ['DLCST', 'DLCST/']:
        name_remote_found_boundboxes_file = \
            join_path_names(source_remote_datadir, 'Others/found_boundingBox_croppedCTinFull.npy')
    else:
        name_remote_found_boundboxes_file = None

    output_datadir = join_path_names(currentdir(), output_datadir)
    name_input_raw_images_path = join_path_names(output_datadir, NAME_RAW_IMAGES_RELPATH)
    name_input_raw_labels_path = join_path_names(output_datadir, NAME_RAW_LABELS_RELPATH)
    name_input_raw_roimasks_path = join_path_names(output_datadir, NAME_RAW_ROIMASKS_RELPATH)
    name_input_reference_files_path = join_path_names(output_datadir, NAME_REFERENCE_FILES_RELPATH)
    name_input_crop_boundboxes_file = join_path_names(output_datadir, NAME_CROP_BOUNDBOXES_FILE)
    name_input_rescale_factors_file = join_path_names(output_datadir, NAME_RESCALE_FACTORS_FILE)

    if args.is_prepare_centrelines:
        name_input_raw_centrelines_path = join_path_names(output_datadir, NAME_RAW_CENTRELINES_RELPATH)
    else:
        name_input_raw_centrelines_path = None

    if args.is_prepare_coarse_airways:
        name_input_raw_coarse_airways_path = join_path_names(output_datadir, NAME_RAW_COARSEAIRWAYS_RELPATH)
    else:
        name_input_raw_coarse_airways_path = None

    if args.source_remote_datadir in ['DLCST', 'DLCST/']:
        name_input_found_boundboxes_file = join_path_names(output_datadir, basename(name_remote_found_boundboxes_file))
    else:
        name_input_found_boundboxes_file = None

    # *****************************************************

    list_calls_all = []

    # 1st: Download data from the cluster
    new_call = ['rsync', '-avr', name_remote_raw_images_path, name_input_raw_images_path]
    list_calls_all.append(new_call)

    new_call = ['rsync', '-avr', name_remote_raw_labels_path, name_input_raw_labels_path]
    list_calls_all.append(new_call)

    new_call = ['rsync', '-avr', name_remote_raw_roimasks_path, name_input_raw_roimasks_path]
    list_calls_all.append(new_call)

    if args.is_prepare_coarse_airways:
        new_call = ['rsync', '-avr', name_remote_raw_coarse_airways_path, name_input_raw_coarse_airways_path]
        list_calls_all.append(new_call)

    if args.source_remote_datadir in ['DLCST', 'DLCST/']:
        new_call = ['rsync', '-avr', name_remote_found_boundboxes_file, name_input_found_boundboxes_file]
        list_calls_all.append(new_call)

    # Iterate over the list and carry out call serially
    for icall in list_calls_all:
        print_call(icall)
        try:
            launch_call(icall)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            message = 'Call failed. Stop pipeline...'
            catch_error_exception(message)
        print("\n")
    # endfor

    # ******************************

    list_calls_all = []

    # 2nd: Decompress (if needed) and prepare the downloaded data
    sublist_calls = create_task_decompress_data(name_input_raw_images_path, args.is_keep_raw_images)
    list_calls_all += sublist_calls

    sublist_calls = create_task_decompress_data(name_input_raw_labels_path, args.is_keep_raw_images)
    list_calls_all += sublist_calls

    sublist_calls = create_task_decompress_data(name_input_raw_roimasks_path, args.is_keep_raw_images)
    list_calls_all += sublist_calls

    if args.is_prepare_coarse_airways:
        sublist_calls = create_task_decompress_data(name_input_raw_coarse_airways_path, args.is_keep_raw_images)
        list_calls_all += sublist_calls

    # ******************************

    # Binarise the input masks for airway and lungs
    if args.is_keep_raw_images:
        name_tempo_binary_labels_path = set_dirname_suffix(name_input_raw_labels_path, 'Binary')

        new_call = ['python3', SCRIPT_BINARISE_MASKS,
                    name_input_raw_labels_path, name_tempo_binary_labels_path,
                    '--type', 'binarise']
        list_calls_all.append(new_call)

        new_sublist_calls = create_task_replace_dirs(name_input_raw_labels_path, name_tempo_binary_labels_path)
        list_calls_all += new_sublist_calls

        # 'is_two_boundboxes_lungs' needs the two labels of ROI masks to compute the two bounding-boxes for each lung
        if not args.is_two_boundboxes_lungs:
            name_tempo_binary_roimasks_path = set_dirname_suffix(name_input_raw_roimasks_path, 'Binary')

            new_call = ['python3', SCRIPT_BINARISE_MASKS,
                        name_input_raw_roimasks_path, name_tempo_binary_roimasks_path,
                        '--type', 'binarise']
            list_calls_all.append(new_call)

            new_sublist_calls = create_task_replace_dirs(name_input_raw_roimasks_path, name_tempo_binary_roimasks_path)
            list_calls_all += new_sublist_calls

    # ******************************

    # Extract the labels for trachea and main bronchi from the coarse airways (AND FILL HOLES INSIDE THE TRACHEA)
    if args.is_prepare_coarse_airways:
        name_tempo_trachea_main_bronchi_path = \
            set_dirname_suffix(name_input_raw_coarse_airways_path, 'TracheaMainBronchi')

        new_call = ['python3', SCRIPT_GET_TRACHEA_MAIN_BRONCHI,
                    name_input_raw_coarse_airways_path, name_tempo_trachea_main_bronchi_path,
                    '--type', 'masklabels', 'fillholes',
                    '--in_mask_labels', '2', '3', '4',
                    '--no_suffix_outname', 'True']
        list_calls_all.append(new_call)

        new_sublist_calls = create_task_replace_dirs(name_input_raw_coarse_airways_path,
                                                     name_tempo_trachea_main_bronchi_path)
        list_calls_all += new_sublist_calls

    # ******************************

    # 3rd (for DLCST data): Extend the raw images from the cropped and flipped format found in the cluster
    if args.source_remote_datadir in ['DLCST', 'DLCST/']:
        name_tempo_extended_labels_path = set_dirname_suffix(name_input_raw_labels_path, 'Extended')
        name_tempo_extended_roimasks_path = set_dirname_suffix(name_input_raw_roimasks_path, 'Extended')

        new_call = ['python3', SCRIPT_EXTEND_CROPPED_IMAGES,
                    name_input_raw_labels_path, name_tempo_extended_labels_path,
                    '--reference_dir', name_input_reference_files_path,
                    '--boundboxes_file', name_input_found_boundboxes_file]
        list_calls_all.append(new_call)

        new_call = ['python3', SCRIPT_EXTEND_CROPPED_IMAGES,
                    name_input_raw_roimasks_path, name_tempo_extended_roimasks_path,
                    '--reference_dir', name_input_reference_files_path,
                    '--boundboxes_file', name_input_found_boundboxes_file]
        list_calls_all.append(new_call)

        sublist_calls = create_task_replace_dirs(name_input_raw_labels_path, name_tempo_extended_labels_path)
        list_calls_all += sublist_calls

        sublist_calls = create_task_replace_dirs(name_input_raw_roimasks_path, name_tempo_extended_roimasks_path)
        list_calls_all += sublist_calls

        if args.is_prepare_coarse_airways:
            name_tempo_extended_coarse_airways_path = set_dirname_suffix(name_input_raw_coarse_airways_path, 'Extended')

            new_call = ['python3', SCRIPT_EXTEND_CROPPED_IMAGES,
                        name_input_raw_coarse_airways_path, name_tempo_extended_coarse_airways_path,
                        '--reference_dir', name_input_reference_files_path,
                        '--boundboxes_file', name_input_found_boundboxes_file]
            list_calls_all.append(new_call)

            new_sublist_calls = create_task_replace_dirs(name_input_raw_coarse_airways_path,
                                                         name_tempo_extended_coarse_airways_path)
            list_calls_all += new_sublist_calls

    # ******************************

    # 4th: Compute the ground-truth centrelines by thinning the ground-truth airways
    if args.is_prepare_centrelines:
        new_call = ['python3', SCRIPT_COMPUTE_CENTRELINES,
                    name_input_raw_labels_path, name_input_raw_centrelines_path,
                    '--type', 'thinning']
        list_calls_all.append(new_call)

    # ******************************

    # 5th: Compute rescaling factors, and rescale the Roi masks to compute the bounding masks
    if args.is_rescale_images:
        name_tempo_rescaled_roi_masks_path = set_dirname_suffix(name_input_raw_roimasks_path, 'Rescaled')

        new_call = ['python3', SCRIPT_CALC_RESCALE_FACTOR_IMAGES,
                    '--datadir', output_datadir,
                    '--fixed_rescale_resol', str(args.fixed_rescale_resol)]
        list_calls_all.append(new_call)

        new_call = ['python3', SCRIPT_RESCALE_ROI_MASKS,
                    name_input_raw_roimasks_path, name_tempo_rescaled_roi_masks_path,
                    '--type', 'rescale_mask',
                    '--rescale_factors_file', name_input_rescale_factors_file,
                    '--reference_dir', name_input_reference_files_path]
        list_calls_all.append(new_call)

        sublist_calls = create_task_replace_dirs(name_input_raw_labels_path, name_tempo_rescaled_roi_masks_path)
        list_calls_all += sublist_calls

    # ******************************

    # 6th: Compute the bounding-boxes around the ROI masks
    if args.is_crop_images:
        new_call = ['python3', SCRIPT_CALC_BOUNDING_BOX_IMAGES,
                    '--datadir', output_datadir,
                    '--size_buffer_in_borders', str(args.size_buffer_in_borders),
                    '--is_two_boundboxes_lungs', str(args.is_two_boundboxes_lungs),
                    '--size_train_images', str(args.size_train_images),
                    '--is_same_size_boundbox_all_images', str(args.is_same_size_boundbox_all_images),
                    '--name_output_boundboxes_file', name_input_crop_boundboxes_file,
                    '--size_fixed_boundbox_all', str(args.size_fixed_boundbox_all)]
        list_calls_all.append(new_call)

    # ******************************

    # 7th: Prepare the data
    new_call = ['python3', SCRIPT_PREPARE_DATA,
                '--datadir', output_datadir,
                '--is_prepare_labels', str(args.is_prepare_labels),
                '--is_input_extra_labels', 'False',
                '--is_binary_train_masks', 'True',
                '--is_mask_region_interest', str(args.is_mask_region_interest),
                '--is_crop_images', str(args.is_crop_images),
                '--is_rescale_images', str(args.is_rescale_images),
                '--is_two_boundboxes_lungs', str(args.is_two_boundboxes_lungs),
                '--name_crop_boundboxes_file', name_input_crop_boundboxes_file,
                '--name_rescale_factors_file', name_input_rescale_factors_file]
    list_calls_all.append(new_call)

    # ******************************

    # Binarise the ROI masks now, after being used to compute the two bounding-boxes for each lung
    if args.is_two_boundboxes_lungs:
        name_tempo_binary_roimasks_path = set_dirname_suffix(name_input_raw_roimasks_path, 'Binary')

        new_call = ['python3', SCRIPT_BINARISE_MASKS,
                    name_input_raw_roimasks_path, name_tempo_binary_roimasks_path,
                    '--type', 'binarise']
        list_calls_all.append(new_call)

        new_sublist_calls = create_task_replace_dirs(name_input_raw_roimasks_path, name_tempo_binary_roimasks_path)
        list_calls_all += new_sublist_calls

    # ******************************

    # Remove all the data not needed anymore
    if args.type_data == 'training':
        new_call = ['rm', '-r', name_input_raw_images_path]
        list_calls_all.append(new_call)

        new_call = ['rm', '-r', name_input_raw_labels_path]
        list_calls_all.append(new_call)

        new_call = ['rm', '-r', name_input_raw_roimasks_path]
        list_calls_all.append(new_call)

        if args.source_remote_datadir in ['DLCST', 'DLCST/']:
            new_call = ['rm', name_input_found_boundboxes_file]
            list_calls_all.append(new_call)

    # ******************************

    # Iterate over the list and carry out call serially
    for icall in list_calls_all:
        print_call(icall)
        try:
            launch_call(icall)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            message = 'Call failed. Stop pipeline...'
            catch_error_exception(message)
        print("\n")
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source_remote_datadir', type=str)
    parser.add_argument('output_datadir', type=str)
    parser.add_argument('--type_data', type=str, default='training')
    parser.add_argument('--size_train_images', type=str2tuple_int, default=SIZE_IN_IMAGES)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--is_crop_images', type=str2bool, default=IS_CROP_IMAGES)
    parser.add_argument('--is_rescale_images', type=str2bool, default=IS_RESCALE_IMAGES)
    parser.add_argument('--is_two_boundboxes_lungs', type=str2bool, default=IS_TWO_BOUNDBOXES_LUNGS)
    args = parser.parse_args()

    if args.type_data == 'training':
        print("Prepare Training data: Processed Images and Labels...")
        args.is_keep_raw_images = False
        args.is_prepare_labels = True
        args.is_prepare_centrelines = False
        args.is_prepare_coarse_airways = False
        if args.is_crop_images:
            args.size_buffer_in_borders = (20, 20, 20)
            args.is_same_size_boundbox_all_images = False
            args.size_fixed_boundbox_all = None

    elif args.type_data == 'testing':
        print("Prepare Testing data: Only Processed Images. Keep raw Images and Labels for testing...")
        args.is_keep_raw_images = True
        args.is_prepare_labels = False
        args.is_prepare_centrelines = True
        args.is_prepare_coarse_airways = True
        if args.is_crop_images:
            args.size_buffer_in_borders = (50, 50, 50)
            args.is_same_size_boundbox_all_images = False
            args.size_fixed_boundbox_all = None

    else:
        message = 'Input param \'type_data\' = \'%s\' not valid, must be inside: \'%s\'...' \
                  % (args.type_data, LIST_TYPE_DATA_AVAIL)
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
