
from common.constant import *
from common.functionutil import *
import subprocess
import traceback
import sys
import argparse


CODEDIR                         = '/home/antonio/Codes/Antonio_repository/AirwaySegmentation/'
SCRIPT_CONVERT_TO_NIFTI         = join_path_names(CODEDIR, 'scripts_util/convert_images_to_nifti.py')
SCRIPT_BINARISE_MASKS           = join_path_names(CODEDIR, 'scripts_util/apply_operation_images.py')
SCRIPT_GET_TRACHEA_MAIN_BRONCHI = join_path_names(CODEDIR, 'scripts_util/apply_operation_images.py')
SCRIPT_COMPUTE_CENTRELINES      = join_path_names(CODEDIR, 'scripts_util/apply_operation_images.py')
SCRIPT_RESCALE_ROI_MASKS        = join_path_names(CODEDIR, 'scripts_util/apply_operation_images.py')
SCRIPT_EXTEND_CROPPED_IMAGES    = join_path_names(CODEDIR, 'scripts_util/specific_Dlcst/extend_cropped_images_fullsize.py')
SCRIPT_CALC_RESCALE_FACTOR_IMAGES=join_path_names(CODEDIR, 'scripts_airway_segmentation/compute_rescalefactor_images.py')
SCRIPT_CALC_BOUNDING_BOX_IMAGES = join_path_names(CODEDIR, 'scripts_airway_segmentation/compute_boundingbox_images.py')
SCRIPT_PREPARE_DATA             = join_path_names(CODEDIR, 'scripts_experiments/prepare_data.py')

CLUSTER_ARCHIVE_DIR = 'agarcia@bigr-app001:/scratch/agarcia/Data/'

LIST_TYPE_DATA_AVAIL = ['training', 'testing']


def print_call(new_call):
    message = ' '.join(new_call)
    print("\n"+ "*" * 100)
    print("<<< Launch: %s >>>" %(message))
    print("*" * 100 +"\n")

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

    #output_datadir = update_dirname(args.output_datadir)
    output_datadir = args.output_datadir
    makedir(output_datadir)


    source_cluster_data_dir   = join_path_names(CLUSTER_ARCHIVE_DIR, args.in_cluster_casedir)

    name_source_raw_images_path   = join_path_names(source_cluster_data_dir, 'CTs/')
    name_source_raw_labels_path   = join_path_names(source_cluster_data_dir, 'Airways/')
    name_source_raw_RoImasks_path = join_path_names(source_cluster_data_dir, 'Lungs/')
    if args.is_prepare_coarse_airways:
        name_source_raw_coarse_airways_path = join_path_names(source_cluster_data_dir, 'CoarseAirways/')
    if args.in_cluster_casedir in ['DLCST', 'DLCST/']:
        name_source_found_bound_boxes_file  = join_path_names(source_cluster_data_dir, 'Others/found_boundingBox_croppedCTinFull.npy')

    output_datadir                  = join_path_names(currentdir(), output_datadir)
    name_input_raw_images_path      = join_path_names(output_datadir, NAME_RAW_IMAGES_RELPATH)
    name_input_raw_labels_path      = join_path_names(output_datadir, NAME_RAW_LABELS_RELPATH)
    name_input_raw_RoImasks_path    = join_path_names(output_datadir, NAME_RAW_ROIMASKS_RELPATH)
    name_input_reference_files_path = join_path_names(output_datadir, NAME_REFERENCE_FILES_RELPATH)
    if args.is_prepare_centrelines:
        name_input_raw_centrelines_path   = join_path_names(output_datadir, NAME_RAW_CENTRELINES_RELPATH)
    if args.is_prepare_coarse_airways:
        name_input_raw_coarse_airways_path= join_path_names(output_datadir, NAME_RAW_COARSEAIRWAYS_RELPATH)
    if args.is_rescale_images:
        name_input_rescale_factors_file   = join_path_names(output_datadir, NAME_RESCALE_FACTOR_FILE)
    if args.in_cluster_casedir in ['DLCST', 'DLCST/']:
        name_input_found_bound_boxes_file = join_path_names(output_datadir, basename(name_source_found_bound_boxes_file))



    list_calls_all = []

    # 1st: Download data from the cluster
    new_call = ['rsync', '-avr', name_source_raw_images_path, name_input_raw_images_path]
    list_calls_all.append(new_call)

    new_call = ['rsync', '-avr', name_source_raw_labels_path, name_input_raw_labels_path]
    list_calls_all.append(new_call)

    new_call = ['rsync', '-avr', name_source_raw_RoImasks_path, name_input_raw_RoImasks_path]
    list_calls_all.append(new_call)

    if args.is_prepare_coarse_airways:
        new_call = ['rsync', '-avr', name_source_raw_coarse_airways_path, name_input_raw_coarse_airways_path]
        list_calls_all.append(new_call)

    if args.in_cluster_casedir in ['DLCST', 'DLCST/']:
        new_call = ['rsync', '-avr', name_source_found_bound_boxes_file, name_input_found_bound_boxes_file]
        list_calls_all.append(new_call)

    # Iterate over the list and carry out call serially
    for icall in list_calls_all:
        print_call(icall)
        try:
            launch_call(icall)
        except Exception as ex:
            traceback.print_exc(file=sys.stdout)
            message = 'Call failed. Stop pipeline...'
            catch_error_exception(message)
        print('\n')
    # endfor



    list_calls_all = []

    # 2nd: Decompress (if needed) and prepare the downloaded data
    sublist_calls = create_task_decompress_data(name_input_raw_images_path, args.is_keep_raw_images)
    list_calls_all += sublist_calls

    sublist_calls = create_task_decompress_data(name_input_raw_labels_path, args.is_keep_raw_images)
    list_calls_all += sublist_calls

    sublist_calls = create_task_decompress_data(name_input_raw_RoImasks_path, args.is_keep_raw_images)
    list_calls_all += sublist_calls

    if args.is_prepare_coarse_airways:
        sublist_calls = create_task_decompress_data(name_input_raw_coarse_airways_path, args.is_keep_raw_images)
        list_calls_all += sublist_calls


    # Binarise the input masks for airway and lungs
    if args.is_keep_raw_images:
        name_tempo_binary_labels_path   = set_dirname_suffix(name_input_raw_labels_path, 'Binary')
        name_tempo_binary_RoImasks_path = set_dirname_suffix(name_input_raw_RoImasks_path, 'Binary')

        new_call = ['python3', SCRIPT_BINARISE_MASKS, name_input_raw_labels_path, name_tempo_binary_labels_path,
                    '--type', 'binarise']
        list_calls_all.append(new_call)

        new_call = ['python3', SCRIPT_BINARISE_MASKS, name_input_raw_RoImasks_path, name_tempo_binary_RoImasks_path,
                    '--type', 'binarise']
        list_calls_all.append(new_call)

        new_sublist_calls = create_task_replace_dirs(name_input_raw_labels_path, name_tempo_binary_labels_path)
        list_calls_all += new_sublist_calls

        new_sublist_calls = create_task_replace_dirs(name_input_raw_RoImasks_path, name_tempo_binary_RoImasks_path)
        list_calls_all += new_sublist_calls

    # Extract the labels for trachea and main bronchii from the coarse airways
    if args.is_prepare_coarse_airways:
        name_tempo_trachea_main_bronchi_path = set_dirname_suffix(name_input_raw_coarse_airways_path, 'TracheaMainBronchi')

        new_call = ['python3', SCRIPT_GET_TRACHEA_MAIN_BRONCHI, name_input_raw_coarse_airways_path, name_tempo_trachea_main_bronchi_path,
                    '--type', 'masklabels',
                    '--in_mask_labels', '2', '3', '4',
                    '--no_suffix_outname', 'True']
        list_calls_all.append(new_call)

        new_sublist_calls = create_task_replace_dirs(name_input_raw_coarse_airways_path, name_tempo_trachea_main_bronchi_path)
        list_calls_all += new_sublist_calls



    # 3rd (for DLCST data): Extend the raw images from the cropped and flipped format found in the cluster
    if args.in_cluster_casedir in ['DLCST', 'DLCST/']:
        name_tempo_extended_labels_path   = set_dirname_suffix(name_input_raw_labels_path, 'Extended')
        name_tempo_extended_RoImasks_path = set_dirname_suffix(name_input_raw_RoImasks_path, 'Extended')

        new_call = ['python3', SCRIPT_EXTEND_CROPPED_IMAGES, name_input_raw_labels_path, name_tempo_extended_labels_path,
                    '--reference_dir', name_input_reference_files_path,
                    '--boundingbox_file', name_input_found_bound_boxes_file]
        list_calls_all.append(new_call)

        new_call = ['python3', SCRIPT_EXTEND_CROPPED_IMAGES, name_input_raw_RoImasks_path, name_tempo_extended_RoImasks_path,
                    '--reference_dir', name_input_reference_files_path,
                    '--boundingbox_file', name_input_found_bound_boxes_file]
        list_calls_all.append(new_call)

        sublist_calls = create_task_replace_dirs(name_input_raw_labels_path, name_tempo_extended_labels_path)
        list_calls_all += sublist_calls

        sublist_calls = create_task_replace_dirs(name_input_raw_RoImasks_path, name_tempo_extended_RoImasks_path)
        list_calls_all += sublist_calls

        if args.is_prepare_coarse_airways:
            name_tempo_extended_coarse_airways_path = set_dirname_suffix(name_input_raw_coarse_airways_path, 'Extended')

            new_call = ['python3', SCRIPT_EXTEND_CROPPED_IMAGES, name_input_raw_coarse_airways_path, name_tempo_extended_coarse_airways_path,
                        '--reference_dir', name_input_reference_files_path,
                        '--boundingbox_file', name_input_found_bound_boxes_file]
            list_calls_all.append(new_call)

            new_sublist_calls = create_task_replace_dirs(name_input_raw_coarse_airways_path, name_tempo_extended_coarse_airways_path)
            list_calls_all += new_sublist_calls



    # 4th: Compute the ground-truth centrelines by thinning the ground-truth airways
    if args.is_prepare_centrelines:
        new_call = ['python3', SCRIPT_COMPUTE_CENTRELINES, name_input_raw_labels_path, name_input_raw_centrelines_path,
                    '--type', 'thinning']
        list_calls_all.append(new_call)



    # 5th: Compute rescaling factors, and rescale the Roi masks to compute the bounding masks
    if args.is_rescale_images:
        name_tempo_rescaled_roi_masks_path = set_dirname_suffix(name_input_raw_RoImasks_path, 'Rescaled')

        new_call = ['python3', SCRIPT_CALC_RESCALE_FACTOR_IMAGES,
                    '--datadir', output_datadir,
                    '--fixed_rescale_resol', str(args.fixed_rescale_resol)]
        list_calls_all.append(new_call)

        new_call = ['python3', SCRIPT_RESCALE_ROI_MASKS, name_input_raw_RoImasks_path, name_tempo_rescaled_roi_masks_path,
                    '--type', 'rescale_mask',
                    '--rescalefactor_file', name_input_rescale_factors_file,
                    '--reference_dir', name_input_reference_files_path]
        list_calls_all.append(new_call)

        sublist_calls = create_task_replace_dirs(name_input_raw_labels_path, name_tempo_rescaled_roi_masks_path)
        list_calls_all += sublist_calls



    # 6th: Compute the bounding-boxes around the Roi masks
    if args.is_crop_images:
        new_call = ['python3', SCRIPT_CALC_BOUNDING_BOX_IMAGES,
                    '--datadir', output_datadir,
                    '--is_two_bounding_box_each_lungs', str(args.is_two_bounding_box_each_lungs),
                    '--size_buffer_in_borders', str(args.size_buffer_in_borders),
                    '--size_train_images', str(args.size_train_images),
                    '--is_same_size_boundbox_all_images', str(args.is_same_size_boundbox_all_images),
                    '--fixed_size_bounding_box', str(args.fixed_size_bounding_box)]
        list_calls_all.append(new_call)



    # 7th: Prepare the data
    new_call = ['python3', SCRIPT_PREPARE_DATA,
                '--datadir', output_datadir,
                '--is_prepare_labels', str(args.is_prepare_labels),
                '--is_input_extra_labels', 'False',
                '--is_binary_train_masks', 'True',
                '--is_mask_region_interest', str(args.is_mask_region_interest),
                '--is_rescale_images', str(args.is_rescale_images),
                '--is_crop_images', str(args.is_crop_images),
                '--is_RoIlabels_multi_RoImasks', str(args.is_two_bounding_box_each_lungs)]
    list_calls_all.append(new_call)



    # Remove all the data not needed anymore
    if args.type_data == 'training':
        new_call = ['rm', '-r', name_input_raw_images_path]
        list_calls_all.append(new_call)

        new_call = ['rm', '-r', name_input_raw_labels_path]
        list_calls_all.append(new_call)

        new_call = ['rm', '-r', name_input_raw_RoImasks_path]
        list_calls_all.append(new_call)

        if args.in_cluster_casedir in ['DLCST', 'DLCST/']:
            new_call = ['rm', name_input_found_bound_boxes_file]
            list_calls_all.append(new_call)



    # Iterate over the list and carry out call serially
    for icall in list_calls_all:
        print_call(icall)
        try:
            launch_call(icall)
        except Exception as ex:
            traceback.print_exc(file=sys.stdout)
            message = 'Call failed. Stop pipeline...'
            catch_error_exception(message)
        print('\n')
    # endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_cluster_casedir', type=str)
    parser.add_argument('output_datadir', type=str)
    parser.add_argument('--type_data', type=str, default='training')
    parser.add_argument('--size_train_images', type=str2tuple_int, default=SIZE_IN_IMAGES)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--is_rescale_images', type=str2bool, default=IS_RESCALE_IMAGES)
    parser.add_argument('--fixed_rescale_resol', type=str2tuple_float, default=FIXED_RESCALE_RESOL)
    parser.add_argument('--is_crop_images', type=str2bool, default=IS_CROP_IMAGES)
    parser.add_argument('--is_two_bounding_box_each_lungs', type=str2bool, default=IS_TWO_BOUNDBOXES_EACH_LUNGS)
    args = parser.parse_args()

    if args.type_data == 'training':
        print("Prepare Training data: Processed Images and Labels...")
        args.is_keep_raw_images       = False
        args.is_prepare_labels        = True
        args.is_prepare_centrelines   = False
        args.is_prepare_coarse_airways= False
        if args.is_crop_images:
            if args.is_two_bounding_box_each_lungs:
                args.size_buffer_in_borders = (0, 0, 0)
                args.is_same_size_boundbox_all_images = True
                args.fixed_size_bounding_box = args.size_train_images
            else:
                args.size_buffer_in_borders = (20, 20, 20)
                args.is_same_size_boundbox_all_images = False
                args.fixed_size_bounding_box = None

    elif args.type_data == 'testing':
        print("Prepare Testing data: Only Processed Images. Keep raw Images and Labels for testing...")
        args.is_keep_raw_images       = True
        args.is_prepare_labels        = False
        args.is_prepare_centrelines   = True
        args.is_prepare_coarse_airways= True
        if args.is_crop_images:
            args.size_buffer_in_borders = (50, 50, 50)
            args.is_same_size_boundbox_all_images = False
            args.fixed_size_bounding_box = None
    else:
        message = 'Input param \'type_data\' = \'%s\' not valid, must be inside: \'%s\'...' % (args.type_data, LIST_TYPE_DATA_AVAIL)
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)