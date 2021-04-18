
from collections import OrderedDict
import numpy as np
import argparse

from common.constant import BASEDIR, NAME_TESTINGDATA_RELPATH, SIZE_IN_IMAGES, PROP_OVERLAP_SLIDING_WINDOW_TEST, \
    IS_SLIDING_WINDOW_IMAGES, IS_RANDOM_WINDOW_IMAGES, TYPE_LOSS, LIST_TYPE_METRICS, IS_VALID_CONVOLUTIONS, \
    IS_MASK_REGION_INTEREST, NAME_TEMPO_POSTERIORS_RELPATH, NAME_REFERENCE_KEYS_PROCIMAGE_FILE, \
    NAME_REFERENCE_KEYS_POSTERIORS_FILE, TYPE_DNNLIB_USED
from common.functionutil import join_path_names, is_exist_file, basename, basename_filenoext, list_files_dir, \
    str2bool, str2list_str, str2tuple_int, str2tuple_float, read_dictionary, read_dictionary_configparams, \
    save_dictionary, save_dictionary_csv, NetworksUtil
from common.exceptionmanager import catch_error_exception, catch_warning_exception
from common.workdirmanager import TrainDirManager
from dataloaders.dataloader_manager import get_train_imagedataloader_1image
from dataloaders.imagefilereader import ImageFileReader
from models.model_manager import get_model_trainer, get_network_checker
from postprocessing.postprocessing_manager import get_images_reconstructor


def func_extract_caseprocname(in_filename: str) -> str:
    return basename_filenoext(in_filename).replace('images_proc', '')


def main(args):

    # SETTINGS
    name_input_images_files = 'images_proc*.nii.gz'
    # name_input_labels_files = 'labels_proc*.nii.gz'
    # name_input_extra_labels_files = 'cenlines_proc*.nii.gz'
    if args.is_save_featmaps_layer:
        name_output_prediction_files = 'featmaps_proc%s_lay-%s_feat%0.2i.nii.gz'
    else:
        name_output_prediction_files = 'probmaps_proc%s.nii.gz'
    # --------

    workdir_manager = TrainDirManager(args.basedir)
    testing_data_path = workdir_manager.get_pathdir_exist(args.testing_datadir)
    in_reference_keys_file = workdir_manager.get_datafile_exist(args.name_input_reference_keys_file)
    output_predictions_path = workdir_manager.get_pathdir_new(args.name_output_predictions_relpath)
    out_reference_keys_file = workdir_manager.get_pathfile_new(args.name_output_reference_keys_file)
    list_test_images_files = list_files_dir(testing_data_path, name_input_images_files)
    indict_reference_keys = read_dictionary(in_reference_keys_file)

    # *****************************************************

    # LOADING MODEL
    print("\nLoad Saved model...")
    print("-" * 30)

    print("Compute Predictions from file: \'%s\'..." % (args.input_model_file))
    model_trainer = get_model_trainer()

    print("Load full model (model weights and description, optimizer, loss and metrics) to restart model...")
    if TYPE_DNNLIB_USED == 'Keras':
        # CHECK WHETHER THIS IS NECESSARY
        model_trainer.create_loss(type_loss=args.type_loss,
                                  is_mask_to_region_interest=args.is_mask_region_interest)
        model_trainer.create_list_metrics(list_type_metrics=args.list_type_metrics,
                                          is_mask_to_region_interest=args.is_mask_region_interest)
    if args.is_backward_compat:
        model_trainer.load_model_full_backward_compat(args.input_model_file)
    else:
        model_trainer.load_model_full(args.input_model_file)

    # model_trainer.summary_model()

    if args.is_save_featmaps_layer:
        print("Compute and store Feature Maps from the Model layer \'%s\'..." % (args.name_layer_save_featmaps))
        network_checker = get_network_checker(args.size_in_images, model_trainer._network)
    else:
        network_checker = None

    size_output_image_model = model_trainer.get_size_output_image_model()

    if args.is_valid_convolutions:
        print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..."
              % (str(args.size_in_images), str(size_output_image_model)))

    if args.is_reconstruct_pred_patches:
        # Create Image Reconstructor
        images_reconstructor = get_images_reconstructor(args.size_in_images,
                                                        is_sliding_window=True,
                                                        prop_overlap_slide_images=args.prop_overlap_sliding_window,
                                                        is_random_window=False,
                                                        num_random_images=0,
                                                        is_transform_rigid=False,
                                                        trans_rigid_params=None,
                                                        is_transform_elastic=False,
                                                        type_trans_elastic='',
                                                        is_nnet_validconvs=args.is_valid_convolutions,
                                                        size_output_images=size_output_image_model,
                                                        is_filter_output_images=args.is_filter_output_network,
                                                        size_filter_output_images=args.size_filter_output_images)
    else:
        images_reconstructor = None

    # *****************************************************

    print("\nCompute Predictions, from \'%s\' files..." % (len(list_test_images_files)))
    print("-" * 30)

    outdict_reference_keys = OrderedDict()

    for ifile, in_image_file in enumerate(list_test_images_files):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        print("Loading data...")
        image_data_loader = \
            get_train_imagedataloader_1image([in_image_file],
                                             size_images=args.size_in_images,
                                             is_sliding_window=args.is_reconstruct_pred_patches,
                                             prop_overlap_slide_images=args.prop_overlap_sliding_window,
                                             is_random_window=False,
                                             num_random_images=0,
                                             is_transform_rigid=False,
                                             trans_rigid_params=None,
                                             is_transform_elastic=False,
                                             type_trans_elastic='',
                                             batch_size=1,
                                             is_shuffle=False,
                                             manual_seed=None)
        print("Loaded \'%s\' files. Total patches generated: \'%s\'..." % (1, len(image_data_loader)))

        # ******************************

        if args.is_save_featmaps_layer:
            print("Evaluate Model feature maps...")
            out_prediction_patches = network_checker.get_feature_maps(image_data_loader, args.name_layer_save_featmaps)
        else:
            print("Evaluate Model...")
            out_prediction_patches = model_trainer.predict(image_data_loader)

        # ******************************

        if args.is_reconstruct_pred_patches:
            print("\nReconstruct full size Prediction from sliding-window image patches...")
            shape_reconstructed_image = ImageFileReader.get_image_size(in_image_file)

            images_reconstructor.initialize_recons_data(shape_reconstructed_image)
            images_reconstructor.initialize_recons_array(out_prediction_patches[0])

            out_prediction_reconstructed = images_reconstructor.compute_full(out_prediction_patches)
        else:
            out_prediction_reconstructed = np.squeeze(out_prediction_patches, axis=(0, -1))

        # ******************************

        # Output predictions
        in_reference_key = indict_reference_keys[basename_filenoext(in_image_file)]
        in_caseproc_name = func_extract_caseprocname(in_image_file)

        if args.is_save_featmaps_layer:
            num_featmaps = out_prediction_reconstructed.shape[-1]
            print("Output model Feature maps (\'%s\' in total)..." % (num_featmaps))

            for ifeat in range(num_featmaps):
                output_prediction_file = join_path_names(output_predictions_path,
                                                         name_output_prediction_files % (in_caseproc_name,
                                                                                         args.name_layer_save_featmaps,
                                                                                         ifeat + 1))
                print("Output: \'%s\', of dims \'%s\'..." % (basename(output_prediction_file),
                                                             out_prediction_reconstructed[..., ifeat].shape))

                ImageFileReader.write_image(output_prediction_file, out_prediction_reconstructed[..., ifeat])

                outdict_reference_keys[basename_filenoext(output_prediction_file)] = basename(in_reference_key)
            # endfor
        else:
            output_prediction_file = join_path_names(output_predictions_path,
                                                     name_output_prediction_files % (in_caseproc_name))
            print("Output: \'%s\', of dims \'%s\'..." % (basename(output_prediction_file),
                                                         out_prediction_reconstructed.shape))

            ImageFileReader.write_image(output_prediction_file, out_prediction_reconstructed)

            outdict_reference_keys[basename_filenoext(output_prediction_file)] = basename(in_reference_key)
    # endfor

    # Save reference keys for predictions
    save_dictionary(out_reference_keys_file, outdict_reference_keys)
    save_dictionary_csv(out_reference_keys_file.replace('.npy', '.csv'), outdict_reference_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model_file', type=str)
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--in_config_file', type=str, default=None)
    parser.add_argument('--testing_datadir', type=str, default=NAME_TESTINGDATA_RELPATH)
    parser.add_argument('--size_in_images', type=str2tuple_int, default=SIZE_IN_IMAGES)
    parser.add_argument('--is_reconstruct_pred_patches', type=str2bool, default=(IS_SLIDING_WINDOW_IMAGES
                                                                                 or IS_RANDOM_WINDOW_IMAGES))
    parser.add_argument('--prop_overlap_sliding_window', type=str2tuple_float, default=PROP_OVERLAP_SLIDING_WINDOW_TEST)
    parser.add_argument('--type_loss', type=str, default=TYPE_LOSS)
    parser.add_argument('--list_type_metrics', type=str2list_str, default=LIST_TYPE_METRICS)
    parser.add_argument('--is_valid_convolutions', type=str2bool, default=IS_VALID_CONVOLUTIONS)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--name_output_predictions_relpath', type=str, default=NAME_TEMPO_POSTERIORS_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--name_output_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_POSTERIORS_FILE)
    parser.add_argument('--is_filter_output_network', type=str2bool, default=False)
    parser.add_argument('--is_save_featmaps_layer', type=str2bool, default=False)
    parser.add_argument('--name_layer_save_featmaps', type=str, default=None)
    parser.add_argument('--is_backward_compat', type=str2bool, default=False)
    args = parser.parse_args()

    if args.in_config_file:
        if not is_exist_file(args.in_config_file):
            message = "Config params file not found: \'%s\'" % (args.in_config_file)
            catch_error_exception(message)
        else:
            input_args_file = read_dictionary_configparams(args.in_config_file)
            print("Set up experiments with parameters from file: \'%s\'" % (args.in_config_file))

            # args.basedir = str(input_args_file['workdir'])
            args.size_in_images = str2tuple_int(input_args_file['size_in_images'])
            args.type_loss = str(input_args_file['type_loss'])
            args.list_type_metrics = str2list_str(input_args_file['list_type_metrics'])
            args.is_valid_convolutions = str2bool(input_args_file['is_valid_convolutions'])
            args.is_mask_region_interest = str2bool(input_args_file['is_mask_region_interest'])
            args.is_reconstruct_pred_patches = str2bool(input_args_file['is_sliding_window_images']) or \
                str2bool(input_args_file['is_random_window_images'])

    if args.is_valid_convolutions and not args.is_filter_output_network:
        message = 'Testing network with non-valid convols: better to filter the network output to reduce border effects'
        catch_warning_exception(message)

    if args.is_filter_output_network:
        print("Using the option to filter the network output to reduce border effects...")
        args.size_filter_output_images = NetworksUtil.calc_size_output_layer_valid_convols(args.size_in_images)
        print("Filtering the output images outside the window: %s..." % (str(args.size_filter_output_images)))
    else:
        args.size_filter_output_images = None

    print("Print input arguments...")
    for key, value in sorted(vars(args).items()):
        print("\'%s\' = %s" % (key, value))

    main(args)
