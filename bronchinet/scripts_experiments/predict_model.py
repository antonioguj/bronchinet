
from common.constant import *
from common.functionutil import *
from common.workdirmanager import TrainDirManager
from dataloaders.dataloader_manager import get_train_imagedataloader_1image
from dataloaders.imagefilereader import ImageFileReader
from models.model_manager import ModelTrainer, VisualModelParams
from postprocessing.postprocessing_manager import get_images_reconstructor
from collections import OrderedDict
import argparse


def func_extract_caseprocname_filename(in_filename: str) -> str:
    return basename_filenoext(in_filename).replace('images_proc', '')



def main(args):
    # ---------- SETTINGS ----------
    name_input_images_files = 'images_proc*.nii.gz'
    name_input_labels_files = 'labels_proc*.nii.gz'
    name_input_extra_labels_files = 'cenlines_proc*.nii.gz'
    if (args.is_save_featmaps_layers):
        name_output_prediction_files = 'featmaps_proc%s_lay-%s_feat%0.2i.nii.gz'
    else:
        name_output_prediction_files = 'probmaps_proc%s.nii.gz'
    # ---------- SETTINGS ----------


    workdir_manager         = TrainDirManager(args.basedir)
    testing_data_path       = workdir_manager.get_pathdir_exist(args.testing_datadir)
    in_reference_keys_file  = workdir_manager.get_datafile_exist(args.name_input_reference_keys_file)
    output_predictions_path = workdir_manager.get_pathdir_new(args.name_output_predictions_relpath)
    out_reference_keys_file = workdir_manager.get_pathfile_new(args.name_output_reference_keys_file)
    list_test_images_files  = list_files_dir(testing_data_path, name_input_images_files)
    indict_reference_keys   = read_dictionary(in_reference_keys_file)


    # LOADING MODEL
    print("\nLoad Saved model...")
    print("-" * 30)

    print("Compute Predictions from file: \'%s\'..." % (args.preds_model_file))
    model_trainer = ModelTrainer()

    print("Load full model (model weights and description, optimizer, loss and metrics) to restart model...")
    if TYPE_DNNLIB_USED == 'Keras':
        # CHECK WHETHER THIS IS NECESSARY
        model_trainer.create_loss(type_loss=args.type_loss,
                                  is_mask_to_region_interest=args.is_mask_region_interest)
        model_trainer.create_list_metrics(list_type_metrics=args.list_type_metrics,
                                          is_mask_to_region_interest=args.is_mask_region_interest)
    if args.is_backward_compat:
        model_trainer.load_model_full_backward_compat(args.preds_model_file)
    else:
        model_trainer.load_model_full(args.preds_model_file)

    #model_trainer.summary_model()

    if (args.is_save_featmaps_layers):
        print("Compute and store Feature Maps from the Model layer \'%s\'..." %(args.name_save_feats_model_layer))
        visual_model_params = VisualModelParams(model_trainer._network, args.size_in_images)

    size_out_image_model = model_trainer.get_size_output_image_model()

    if args.is_valid_convolutions:
        print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..." % (str(args.size_in_images),
                                                                                               str(size_out_image_model)))

    if args.is_reconstruct_preds_from_patches:
        # Create Image Reconstructor
        images_reconstructor = get_images_reconstructor(args.size_in_images,
                                                        use_sliding_window_images=True,
                                                        prop_overlap_slide_window=PROP_OVERLAP_SLIDING_WINDOW_TESTING,
                                                        use_random_window_images=False,
                                                        num_random_patches_epoch=0,
                                                        use_transform_rigid_images=False,
                                                        use_transform_elasticdeform_images=False,
                                                        is_output_nnet_validconvs=args.is_valid_convolutions,
                                                        size_output_image=size_out_image_model,
                                                        is_filter_output_nnet=IS_FILTER_PRED_PROBMAPS,
                                                        prop_filter_output_nnet=PROP_VALID_OUTPUT_NNET)

    print("\nCompute Predictiions, from \'%s\' files..." % (len(list_test_images_files)))
    print("-" * 30)

    outdict_reference_keys = OrderedDict()

    for ifile, in_image_file in enumerate(list_test_images_files):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        print("Loading data...")
        image_data_loader = get_train_imagedataloader_1image([in_image_file],
                                                             size_in_images=args.size_in_images,
                                                             use_sliding_window_images=args.is_reconstruct_preds_from_patches,
                                                             prop_overlap_slide_window=PROP_OVERLAP_SLIDING_WINDOW_TESTING,
                                                             use_transform_rigid_images=False,
                                                             use_transform_elasticdeform_images=False,
                                                             use_random_window_images=False,
                                                             num_random_patches_epoch=0,
                                                             batch_size=1,
                                                             is_shuffle=False)
        print("Loaded \'%s\' files. Total batches generated: %s..." % (1, len(image_data_loader)))


        if (args.is_save_featmaps_layers):
            print("Evaluate Model feature maps...")
            out_prediction_batches = visual_model_params.get_feature_maps(image_data_loader, args.name_save_feats_model_layer)
        else:
            print("Evaluate Model...")
            out_prediction_batches = model_trainer.predict(image_data_loader)


        if args.is_reconstruct_preds_from_patches:
            print("\nReconstruct full size Prediction from image patches...")
            out_shape_reconstructed_image = ImageFileReader.get_image_size(in_image_file)
            images_reconstructor.update_image_data(out_shape_reconstructed_image)

            out_prediction_reconstructed = images_reconstructor.compute(out_prediction_batches)
        else:
            out_prediction_reconstructed = np.squeeze(out_prediction_batches, axis=(0, -1))


        # Output predictions
        in_reference_key = indict_reference_keys[basename_filenoext(in_image_file)]
        in_caseprocname_file = func_extract_caseprocname_filename(in_image_file)

        if (args.is_save_featmaps_layers):
            num_featmaps = out_prediction_reconstructed.shape[-1]
            print("Output model Feature maps (\'%s\' in total)..." %(num_featmaps))

            for ifeatmap in range(num_featmaps):
                output_prediction_file = join_path_names(output_predictions_path, name_output_prediction_files % (in_caseprocname_file, args.name_save_feats_model_layer, ifeatmap+1))
                print("Output: \'%s\', of dims \'%s\'..." %(basename(output_prediction_file), out_prediction_reconstructed[...,ifeatmap].shape))

                ImageFileReader.write_image(output_prediction_file, out_prediction_reconstructed[..., ifeatmap])

                outdict_reference_keys[basename_filenoext(output_prediction_file)] = basename(in_reference_key)
            # endfor
        else:
            output_prediction_file = join_path_names(output_predictions_path, name_output_prediction_files % (in_caseprocname_file))
            print("Output: \'%s\', of dims \'%s\'..." % (basename(output_prediction_file), out_prediction_reconstructed.shape))

            ImageFileReader.write_image(output_prediction_file, out_prediction_reconstructed)

            outdict_reference_keys[basename_filenoext(output_prediction_file)] = basename(in_reference_key)
    # endfor

    # Save reference keys for predictions
    save_dictionary(out_reference_keys_file, outdict_reference_keys)
    save_dictionary_csv(out_reference_keys_file.replace('.npy', '.csv'), outdict_reference_keys)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('preds_model_file', type=str)
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--in_config_file', type=str, default=None)
    parser.add_argument('--testing_datadir', type=str, default=NAME_TESTINGDATA_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--name_output_predictions_relpath', type=str, default=NAME_TEMPO_POSTERIORS_RELPATH)
    parser.add_argument('--name_output_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_POSTERIORS_FILE)
    parser.add_argument('--size_in_images', type=str2tuple_int, default=SIZE_IN_IMAGES)
    parser.add_argument('--is_reconstruct_preds_from_patches', type=str2bool, default=(USE_SLIDING_WINDOW_IMAGES or USE_RANDOM_WINDOW_IMAGES))
    parser.add_argument('--type_loss', type=str, default=TYPE_LOSS)
    parser.add_argument('--list_type_metrics', type=str2list_str, default=LIST_TYPE_METRICS)
    parser.add_argument('--is_valid_convolutions', type=str2bool, default=IS_VALID_CONVOLUTIONS)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--is_save_featmaps_layers', type=str2bool, default=IS_SAVE_FEATMAPS_LAYERS)
    parser.add_argument('--name_save_feats_model_layer', type=str, default=NAME_SAVE_FEATS_MODEL_LAYER)
    parser.add_argument('--is_backward_compat', type=str2bool, default=False)
    args = parser.parse_args()

    if args.in_config_file:
        if not is_exist_file(args.in_config_file):
            message = "Config params file not found: \'%s\'" % (args.in_config_file)
            catch_error_exception(message)
        else:
            input_args_file = read_dictionary_configparams(args.in_config_file)
        print("Set up experiments with parameters from file: \'%s\'" %(args.in_config_file))
        #args.basedir                        = str(input_args_file['workdir'])
        args.size_in_images                 = str2tuple_int(input_args_file['size_in_images'])
        args.type_loss                      = str(input_args_file['type_loss'])
        args.list_type_metrics              = str2list_str(input_args_file['list_type_metrics'])
        args.is_mask_region_interest        = str2bool(input_args_file['is_mask_region_interest'])
        args.is_valid_convolutions          = str2bool(input_args_file['is_valid_convolutions'])
        args.is_reconstruct_preds_from_patches = str2bool(input_args_file['use_sliding_window_images']) or str2bool(input_args_file['use_random_window_images'])

    print("Print input arguments...")
    for key, value in sorted(vars(args).items()):
        print("\'%s\' = %s" %(key, value))

    main(args)