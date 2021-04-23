
from typing import List, Dict
from collections import OrderedDict
import numpy as np
import argparse

from common.constant import BASEDIR, NAME_MODELSRUN_RELPATH, SIZE_IN_IMAGES, NAME_TRAININGDATA_RELPATH, \
    NAME_VALIDATIONDATA_RELPATH, MAX_TRAIN_IMAGES, MAX_VALID_IMAGES, TYPE_NETWORK, NET_NUM_FEATMAPS, TYPE_OPTIMIZER, \
    LEARN_RATE, TYPE_LOSS, WEIGHT_COMBINED_LOSS, LIST_TYPE_METRICS, BATCH_SIZE, NUM_EPOCHS, IS_VALID_CONVOLUTIONS, \
    IS_MASK_REGION_INTEREST, IS_GENERATE_PATCHES, TYPE_GENERATE_PATCHES, PROP_OVERLAP_SLIDE_WINDOW, \
    NUM_RANDOM_PATCHES_EPOCH, IS_TRANSFORM_IMAGES, TYPE_TRANSFORM_IMAGES, TRANS_RIGID_ROTATION_RANGE, \
    TRANS_RIGID_SHIFT_RANGE, TRANS_RIGID_FLIP_DIRS, TRANS_RIGID_ZOOM_RANGE, TRANS_RIGID_FILL_MODE, \
    FREQ_SAVE_CHECK_MODELS, FREQ_VALIDATE_MODELS, IS_USE_VALIDATION_DATA, IS_SHUFFLE_TRAINDATA, MANUAL_SEED_TRAIN, \
    NAME_REFERENCE_KEYS_PROCIMAGE_FILE, NAME_LOSSHISTORY_FILE, NAME_CONFIG_PARAMS_FILE, NAME_TRAINDATA_LOGFILE, \
    NAME_VALIDDATA_LOGFILE, TYPE_DNNLIB_USED
from common.functionutil import join_path_names, is_exist_file, update_filename, basename, basename_filenoext, \
    list_files_dir, get_substring_filename, str2bool, str2int, str2float, str2list_str, str2tuple_bool, str2tuple_int,\
    str2tuple_float, read_dictionary, read_dictionary_configparams, save_dictionary_configparams
from common.exceptionmanager import catch_error_exception
from common.workdirmanager import TrainDirManager
from dataloaders.dataloader_manager import get_train_imagedataloader_2images
from models.model_manager import get_model_trainer
if TYPE_DNNLIB_USED == 'Pytorch':
    from models.pytorch.modeltrainer import NAME_SAVEDMODEL_LAST
elif TYPE_DNNLIB_USED == 'Keras':
    from models.keras.modeltrainer import NAME_SAVEDMODEL_LAST

LIST_AVAIL_GENERATE_PATCHES_TRAINING = ['slide_window', 'random_window']


def write_train_valid_data_logfile(out_filename: str,
                                   list_data_files: List[str],
                                   dict_reference_keys: Dict[str, str],
                                   type_data: str
                                   ) -> None:
    if is_exist_file(out_filename):
        out_filename = update_filename(out_filename)
    print("Write log for train / valid data in file: \'%s\'..." % (out_filename))

    with open(out_filename, 'w') as fout:
        fout.write('Total of %s %s files\n' % (len(list_data_files), type_data))
        fout.write('---------------------------------\n')
        for in_file in list_data_files:
            fout.write('%s -> (%s)\n' % (basename(in_file), dict_reference_keys[basename_filenoext(in_file)]))


def get_restart_epoch_from_loss_history_file(loss_filename: str) -> int:
    data_file = np.genfromtxt(loss_filename, dtype=str, delimiter=' ')
    last_epoch_file = int(data_file[-1, 0])     # retrieve the last epoch stored
    return last_epoch_file


def get_restart_epoch_from_model_filename(model_filename: str) -> int:
    epoch_filename = get_substring_filename('e[0-9]+', model_filename)
    epoch_filename = int(epoch_filename[1:])    # remove the leading 'e'
    return epoch_filename


def main(args):

    # SETTINGS
    name_input_images_files = 'images_proc*.nii.gz'
    name_input_labels_files = 'labels_proc*.nii.gz'
    # name_input_extra_labels_files = 'cenlines_proc*.nii.gz'
    # --------

    workdir_manager = TrainDirManager(args.basedir)
    training_data_path = workdir_manager.get_pathdir_exist(args.training_datadir)
    in_reference_keys_file = workdir_manager.get_datafile_exist(args.name_reference_keys_file)
    list_train_images_files = list_files_dir(training_data_path, name_input_images_files)[0:args.max_train_images]
    list_train_labels_files = list_files_dir(training_data_path, name_input_labels_files)[0:args.max_train_images]
    indict_reference_keys = read_dictionary(in_reference_keys_file)

    if args.is_restart:
        models_path = workdir_manager.get_pathdir_exist(args.modelsdir)
    else:
        models_path = workdir_manager.get_pathdir_update(args.modelsdir)

    if args.is_use_validation_data:
        validation_data_path = workdir_manager.get_pathdir_exist(args.validation_datadir)
        list_valid_images_files = list_files_dir(validation_data_path, name_input_images_files)[0:args.max_valid_images]
        list_valid_labels_files = list_files_dir(validation_data_path, name_input_labels_files)[0:args.max_valid_images]
    else:
        list_valid_images_files = None
        list_valid_labels_files = None

    # write out config parameters in config file
    out_config_params_file = join_path_names(models_path, NAME_CONFIG_PARAMS_FILE)
    if not args.is_restart or (args.is_restart and args.is_restart_only_weights):
        print("Write configuration parameters in file: \'%s\'..." % (out_config_params_file))
        dict_in_args = OrderedDict(sorted(vars(args).items()))
        dict_in_args.pop('dict_trans_rigid_parameters')     # remove this in config file
        save_dictionary_configparams(out_config_params_file, dict_in_args)

    # write out logs with the training and validation files used
    out_traindata_logfile = join_path_names(models_path, NAME_TRAINDATA_LOGFILE)
    write_train_valid_data_logfile(out_traindata_logfile, list_train_images_files,
                                   indict_reference_keys, 'training')
    if args.is_use_validation_data:
        out_validdata_logfile = join_path_names(models_path, NAME_VALIDDATA_LOGFILE)
        write_train_valid_data_logfile(out_validdata_logfile, list_valid_images_files,
                                       indict_reference_keys, 'validation')

    # *****************************************************

    # BUILDING MODEL
    print("\nBuilding model...")
    print("-" * 30)

    model_trainer = get_model_trainer()

    if not args.is_restart or (args.is_restart and args.is_restart_only_weights):
        model_trainer.create_network(type_network=args.type_network,
                                     size_image_in=args.size_in_images,
                                     num_featmaps_in=args.net_num_featmaps,
                                     num_channels_in=1,
                                     num_classes_out=1,
                                     is_use_valid_convols=args.is_valid_convolutions,
                                     manual_seed=args.manual_seed_train)
        model_trainer.create_optimizer(type_optimizer=args.type_optimizer,
                                       learn_rate=args.learn_rate)
        model_trainer.create_loss(type_loss=args.type_loss,
                                  is_mask_to_region_interest=args.is_mask_region_interest,
                                  weight_combined_loss=args.weight_combined_loss)
        model_trainer.create_list_metrics(list_type_metrics=args.list_type_metrics,
                                          is_mask_to_region_interest=args.is_mask_region_interest)
        model_trainer.finalise_model()

    if args.is_restart:
        model_restart_file = join_path_names(models_path, args.restart_file)
        print("Restart Model from file: \'%s\'..." % (model_restart_file))

        if args.is_restart_only_weights:
            print("Load only saved weights to restart model...")
            model_trainer.load_model_only_weights(model_restart_file)
        else:
            print("Load full model (model weights and description, optimizer, loss and metrics) to restart model...")
            if TYPE_DNNLIB_USED == 'Keras':
                # CHECK WHETHER THIS IS NECESSARY
                model_trainer.create_loss(type_loss=args.type_loss,
                                          is_mask_to_region_interest=args.is_mask_region_interest,
                                          weight_combined_loss=args.weight_combined_loss)
                model_trainer.create_list_metrics(list_type_metrics=args.list_type_metrics,
                                                  is_mask_to_region_interest=args.is_mask_region_interest)
            if args.is_backward_compat:
                model_trainer.load_model_full_backward_compat(model_restart_file)
            else:
                model_trainer.load_model_full(model_restart_file)
    else:
        model_restart_file = None

    model_trainer.create_callbacks(models_path=models_path,
                                   losshist_filename=NAME_LOSSHISTORY_FILE,
                                   is_validation_data=args.is_use_validation_data,
                                   freq_save_check_model=args.freq_save_check_models,
                                   freq_validate_model=args.freq_validate_models,
                                   is_restart_model=args.is_restart)
    # model_trainer.summary_model()

    # *****************************************************

    # LOADING DATA
    print("\nLoading data...")
    print("-" * 30)

    size_output_image_model = model_trainer.get_size_output_image_model()

    if args.is_valid_convolutions:
        print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..."
              % (str(args.size_in_images), str(size_output_image_model)))

    print("\nLoading Training data...")
    training_data_loader = \
        get_train_imagedataloader_2images(list_train_images_files,
                                          list_train_labels_files,
                                          size_images=args.size_in_images,
                                          is_generate_patches=args.is_generate_patches,
                                          type_generate_patches=args.type_generate_patches,
                                          prop_overlap_slide_images=args.prop_overlap_slide_window,
                                          num_random_images=args.num_random_patches_epoch,
                                          is_transform_images=args.is_transform_images,
                                          type_transform_images=args.type_transform_images,
                                          trans_rigid_params=args.dict_trans_rigid_parameters,
                                          is_nnet_validconvs=args.is_valid_convolutions,
                                          size_output_images=size_output_image_model,
                                          batch_size=args.batch_size,
                                          is_shuffle=args.is_shuffle_traindata,
                                          manual_seed=args.manual_seed_train)
    print("Loaded \'%s\' files. Total batches generated: \'%s\'..."
          % (len(list_train_images_files), len(training_data_loader)))

    if args.is_use_validation_data:
        print("\nLoading Validation data...")
        if args.is_generate_patches:
            type_generate_patches_validation = 'slicing'
        else:
            type_generate_patches_validation = ''

        validation_data_loader = \
            get_train_imagedataloader_2images(list_valid_images_files,
                                              list_valid_labels_files,
                                              size_images=args.size_in_images,
                                              is_generate_patches=args.is_generate_patches,
                                              type_generate_patches=type_generate_patches_validation,
                                              prop_overlap_slide_images=args.prop_overlap_slide_window,
                                              num_random_images=0,
                                              is_transform_images=False,
                                              type_transform_images='',
                                              trans_rigid_params=None,
                                              is_nnet_validconvs=args.is_valid_convolutions,
                                              size_output_images=size_output_image_model,
                                              batch_size=args.batch_size,
                                              is_shuffle=args.is_shuffle_traindata,
                                              manual_seed=args.manual_seed_train)
        print("Loaded \'%s\' files. Total batches generated: \'%s\'..."
              % (len(list_valid_images_files), len(validation_data_loader)))
    else:
        validation_data_loader = None

    # *****************************************************

    # TRAINING MODEL
    print("\nTraining model...")
    print("-" * 30)

    if args.is_restart:
        if 'last' in basename(model_restart_file):
            loss_history_filename = join_path_names(models_path, NAME_LOSSHISTORY_FILE)
            restart_epoch = get_restart_epoch_from_loss_history_file(loss_history_filename)
        else:
            restart_epoch = get_restart_epoch_from_model_filename(model_restart_file)

        print("Restarting training from epoch \'%s\'..." % (restart_epoch))
        initial_epoch = restart_epoch
        args.num_epochs += initial_epoch
    else:
        initial_epoch = 0

    model_trainer.train(train_data_loader=training_data_loader,
                        valid_data_loader=validation_data_loader,
                        num_epochs=args.num_epochs,
                        max_steps_epoch=args.max_steps_epoch,
                        initial_epoch=initial_epoch,
                        is_shuffle_data=args.is_shuffle_traindata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--modelsdir', type=str, default=NAME_MODELSRUN_RELPATH)
    parser.add_argument('--in_config_file', type=str, default=None)
    parser.add_argument('--size_in_images', type=str2tuple_int, default=SIZE_IN_IMAGES)
    parser.add_argument('--training_datadir', type=str, default=NAME_TRAININGDATA_RELPATH)
    parser.add_argument('--validation_datadir', type=str, default=NAME_VALIDATIONDATA_RELPATH)
    parser.add_argument('--max_train_images', type=str2int, default=MAX_TRAIN_IMAGES)
    parser.add_argument('--max_valid_images', type=str2int, default=MAX_VALID_IMAGES)
    parser.add_argument('--type_network', type=str, default=TYPE_NETWORK)
    parser.add_argument('--net_num_featmaps', type=str2int, default=NET_NUM_FEATMAPS)
    parser.add_argument('--type_optimizer', type=str, default=TYPE_OPTIMIZER)
    parser.add_argument('--learn_rate', type=str2float, default=LEARN_RATE)
    parser.add_argument('--type_loss', type=str, default=TYPE_LOSS)
    parser.add_argument('--weight_combined_loss', type=str2float, default=WEIGHT_COMBINED_LOSS)
    parser.add_argument('--list_type_metrics', type=str2list_str, default=LIST_TYPE_METRICS)
    parser.add_argument('--batch_size', type=str2int, default=BATCH_SIZE)
    parser.add_argument('--num_epochs', type=str2int, default=NUM_EPOCHS)
    parser.add_argument('--max_steps_epoch', type=str2int, default=None)
    parser.add_argument('--is_valid_convolutions', type=str2bool, default=IS_VALID_CONVOLUTIONS)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--is_generate_patches', type=str2bool, default=IS_GENERATE_PATCHES)
    parser.add_argument('--type_generate_patches', type=str, default=TYPE_GENERATE_PATCHES)
    parser.add_argument('--prop_overlap_slide_window', type=str2tuple_float, default=PROP_OVERLAP_SLIDE_WINDOW)
    parser.add_argument('--num_random_patches_epoch', type=str2int, default=NUM_RANDOM_PATCHES_EPOCH)
    parser.add_argument('--is_transform_images', type=str2bool, default=IS_TRANSFORM_IMAGES)
    parser.add_argument('--type_transform_images', type=str, default=TYPE_TRANSFORM_IMAGES)
    parser.add_argument('--trans_rigid_rotation_range', type=str2tuple_float, default=TRANS_RIGID_ROTATION_RANGE)
    parser.add_argument('--trans_rigid_shift_range', type=str2tuple_float, default=TRANS_RIGID_SHIFT_RANGE)
    parser.add_argument('--trans_rigid_flip_dirs', type=str2tuple_bool, default=TRANS_RIGID_FLIP_DIRS)
    parser.add_argument('--trans_rigid_zoom_range', type=str2float, default=TRANS_RIGID_ZOOM_RANGE)
    parser.add_argument('--trans_rigid_fill_mode', type=str, default=TRANS_RIGID_FILL_MODE)
    parser.add_argument('--freq_save_check_models', type=str2int, default=FREQ_SAVE_CHECK_MODELS)
    parser.add_argument('--freq_validate_models', type=str2int, default=FREQ_VALIDATE_MODELS)
    parser.add_argument('--is_use_validation_data', type=str2bool, default=IS_USE_VALIDATION_DATA)
    parser.add_argument('--is_shuffle_traindata', type=str2bool, default=IS_SHUFFLE_TRAINDATA)
    parser.add_argument('--manual_seed_train', type=str2int, default=MANUAL_SEED_TRAIN)
    parser.add_argument('--name_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--is_restart', type=str2bool, default=False)
    parser.add_argument('--restart_file', type=str, default=NAME_SAVEDMODEL_LAST)
    parser.add_argument('--is_restart_only_weights', type=str2bool, default=False)
    parser.add_argument('--is_backward_compat', type=str2bool, default=False)
    args = parser.parse_args()

    if (args.is_restart and not args.is_restart_only_weights) and not args.in_config_file:
        args.in_config_file = join_path_names(args.modelsdir, NAME_CONFIG_PARAMS_FILE)
        print("Restarting model: input config file is not given. Use the default path: %s" % (args.in_config_file))

    if args.in_config_file:
        if not is_exist_file(args.in_config_file):
            message = "Config params file not found: \'%s\'..." % (args.in_config_file)
            catch_error_exception(message)
        else:
            input_args_file = read_dictionary_configparams(args.in_config_file)
            print("Set up experiments with parameters from file: \'%s\'" % (args.in_config_file))

            # args.basedir = str(input_args_file['basedir'])
            args.size_in_images = str2tuple_int(input_args_file['size_in_images'])
            args.training_datadir = str(input_args_file['training_datadir'])
            args.validation_datadir = str(input_args_file['validation_datadir'])
            args.max_train_images = str2int(input_args_file['max_train_images'])
            args.max_valid_images = str2int(input_args_file['max_valid_images'])
            args.type_network = str(input_args_file['type_network'])
            args.net_num_featmaps = str2int(input_args_file['net_num_featmaps'])
            args.type_optimizer = str(input_args_file['type_optimizer'])
            args.learn_rate = str2float(input_args_file['learn_rate'])
            args.type_loss = str(input_args_file['type_loss'])
            args.weight_combined_loss = str2float(input_args_file['weight_combined_loss'])
            args.list_type_metrics = str2list_str(input_args_file['list_type_metrics'])
            args.batch_size = str2int(input_args_file['batch_size'])
            args.num_epochs = str2int(input_args_file['num_epochs'])
            args.max_steps_epoch = None  # CHECK THIS OUT!
            args.is_valid_convolutions = str2bool(input_args_file['is_valid_convolutions'])
            args.is_mask_region_interest = str2bool(input_args_file['is_mask_region_interest'])
            args.is_generate_patches = str2bool(input_args_file['is_generate_patches'])
            args.type_generate_patches = str(input_args_file['type_generate_patches'])
            args.prop_overlap_slide_window = str2tuple_float(input_args_file['prop_overlap_slide_window'])
            args.num_random_patches_epoch = str2int(input_args_file['num_random_patches_epoch'])
            args.is_transform_images = str2bool(input_args_file['is_transform_images'])
            args.type_transform_images = str(input_args_file['type_transform_images'])
            # args.trans_rigid_rotation_range = str2tuple_float(input_args_file['trans_rigid_rotation_range'])
            # args.trans_rigid_shift_range = str2tuple_float(input_args_file['trans_rigid_shift_range'])
            # args.trans_rigid_flip_dirs = str2tuple_bool(input_args_file['trans_rigid_flip_dirs'])
            # args.trans_rigid_zoom_range = str2float(input_args_file['trans_rigid_zoom_range'])
            # args.trans_rigid_fill_mode = str(input_args_file['trans_rigid_fill_mode'])
            # args.freq_save_check_models = str2int(input_args_file['freq_save_check_models'])
            # args.freq_validate_models = str2int(input_args_file['freq_validate_models'])
            # args.is_use_validation_data = str2bool(input_args_file['is_use_validation_data'])
            # args.is_shuffle_traindata = str2bool(input_args_file['is_shuffle_traindata'])
            args.name_reference_keys_file = str(input_args_file['name_reference_keys_file'])

    if args.type_generate_patches not in LIST_AVAIL_GENERATE_PATCHES_TRAINING:
        message = 'Type of Generate Patches not possible for training: \'%s\'. Options available: \'%s\'' \
                  % (args.type_generate_patches, LIST_AVAIL_GENERATE_PATCHES_TRAINING)
        catch_error_exception(message)

    args.dict_trans_rigid_parameters = {'rotation_range': args.trans_rigid_rotation_range,
                                        'shift_range': args.trans_rigid_shift_range,
                                        'flip_dirs': args.trans_rigid_flip_dirs,
                                        'zoom_range': args.trans_rigid_zoom_range,
                                        'fill_mode': args.trans_rigid_fill_mode}

    print("Print input arguments...")
    for key, value in sorted(vars(args).items()):
        print("\'%s\' = %s" % (key, value))

    main(args)
