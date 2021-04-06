
from typing import List, Dict
from collections import OrderedDict
import numpy as np
import argparse

from common.constant import BASEDIR, SIZE_IN_IMAGES, NAME_TRAININGDATA_RELPATH, NAME_VALIDATIONDATA_RELPATH, \
    MAX_TRAIN_IMAGES, MAX_VALID_IMAGES, BATCH_SIZE, NUM_EPOCHS, TYPE_NETWORK, NET_NUM_LEVELS, NET_NUM_FEATMAPS, \
    TYPE_OPTIMIZER, LEARN_RATE, TYPE_LOSS, WEIGHT_COMBINED_LOSS, LIST_TYPE_METRICS, IS_VALID_CONVOLUTIONS, \
    IS_MASK_REGION_INTEREST, USE_SLIDING_WINDOW_IMAGES, PROP_OVERLAP_SLIDING_WINDOW, USE_RANDOM_WINDOW_IMAGES, \
    NUM_RANDOM_PATCHES_EPOCH, USE_TRANSFORM_RIGID_IMAGES, USE_TRANSFORM_VALIDATION_DATA, USE_VALIDATION_DATA, \
    USE_TRANSFORM_ELASTIC_IMAGES, FREQ_SAVE_INTER_MODELS, FREQ_VALIDATE_MODEL, NAME_LOSSHISTORY_FILE, \
    NAME_CONFIG_PARAMS_FILE, NAME_TRAINDATA_LOGFILE, NAME_VALIDDATA_LOGFILE, NAME_REFERENCE_KEYS_PROCIMAGE_FILE, \
    MANUAL_SEED_TRAIN, IS_SHUFFLE_TRAINDATA, WRITE_OUT_DESC_MODEL_TEXT, NAME_DESCRIPT_MODEL_LOGFILE, TYPE_DNNLIB_USED,\
    IS_RESTART_ONLY_WEIGHTS, IS_MERGE_TWO_IMAGES_AS_CHANNELS, LAYERS_VGG16_LOSS_PERCEPTUAL, \
    WEIGHTS_VGG16_LOSS_PERCEPTUAL, PROP_REDUSIZE_VGG16_LOSS_PERCEPTUAL
if TYPE_DNNLIB_USED == 'Pytorch':
    from common.constant import NAME_SAVEDMODEL_LAST_TORCH as NAME_SAVEDMODEL_LAST
elif TYPE_DNNLIB_USED == 'Keras':
    from common.constant import NAME_SAVEDMODEL_LAST_KERAS as NAME_SAVEDMODEL_LAST
from common.functionutil import join_path_names, is_exist_file, update_filename, basename, basename_filenoext, \
    list_files_dir, get_substring_filename, str2bool, str2int, str2list_str, str2tuple_int, str2tuple_float, \
    read_dictionary, read_dictionary_configparams, save_dictionary_configparams, str2list_float
from common.exceptionmanager import catch_error_exception
from common.workdirmanager import TrainDirManager
from dataloaders.dataloader_manager import get_train_imagedataloader_2images
from models.model_manager import ModelTrainer


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

    if args.is_restart_model:
        models_path = workdir_manager.get_pathdir_exist(args.modelsdir)
    else:
        models_path = workdir_manager.get_pathdir_update(args.modelsdir)

    if USE_VALIDATION_DATA:
        validation_data_path = workdir_manager.get_pathdir_exist(args.validation_datadir)
        list_valid_images_files = list_files_dir(validation_data_path, name_input_images_files)[0:args.max_valid_images]
        list_valid_labels_files = list_files_dir(validation_data_path, name_input_labels_files)[0:args.max_valid_images]
        if not list_valid_images_files or not list_valid_labels_files:
            use_validation_data = False
            print("No validation data used for Training the model...")
        else:
            use_validation_data = True
    else:
        use_validation_data = False

    if args.is_load_augmented_images_per_label:
        if len(list_train_images_files) != args.num_augmented_images_per_label * len(list_train_labels_files):
            message = "Num loaded augmented images (for training) per label does not correspond " \
                      "to the input \'num_augmented_images_per_label\'..."
            catch_error_exception(message)

    # write out logs with the training and validation files files used
    out_traindata_logfile = join_path_names(models_path, NAME_TRAINDATA_LOGFILE)
    write_train_valid_data_logfile(out_traindata_logfile, list_train_images_files,
                                   indict_reference_keys, 'training')

    if use_validation_data:
        out_validdata_logfile = join_path_names(models_path, NAME_VALIDDATA_LOGFILE)
        write_train_valid_data_logfile(out_validdata_logfile, list_valid_images_files,
                                       indict_reference_keys, 'validation')

    # write out experiment parameters in config file
    out_config_params_file = join_path_names(models_path, NAME_CONFIG_PARAMS_FILE)
    if not args.is_restart_model or (args.is_restart_model and args.is_restart_only_weights):
        print("Write configuration parameters in file: \'%s\'..." % (out_config_params_file))
        dict_args = OrderedDict(sorted(vars(args).items()))
        save_dictionary_configparams(out_config_params_file, dict_args)

    # *****************************************************

    # BUILDING MODEL
    print("\nBuilding model...")
    print("-" * 30)

    model_trainer = ModelTrainer()

    if not args.is_restart_model or (args.is_restart_model and args.is_restart_only_weights):
        num_channels_model = 2 if args.is_input_images_two_channels else 1
        model_trainer.create_network(type_network=args.type_network,
                                     size_image_in=args.size_in_images,
                                     num_levels=args.net_num_levels,
                                     num_featmaps_in=args.net_num_featmaps,
                                     num_channels_in=num_channels_model,
                                     num_classes_out=1,
                                     is_use_valid_convols=args.is_valid_convolutions,
                                     manual_seed=args.manual_seed_train)
        model_trainer.create_optimizer(type_optimizer=args.type_optimizer,
                                       learn_rate=args.learn_rate)
        model_trainer.create_loss(type_loss=args.type_loss,
                                  is_mask_to_region_interest=args.is_mask_region_interest,
                                  size_image_in=args.size_in_images,
                                  weight_combined_loss=args.weight_combined_loss,
                                  layers_vgg16_loss_perceptual=args.layers_vgg16_loss_perceptual,
                                  weights_vgg16_loss_perceptual=args.weights_vgg16_loss_perceptual,
                                  prop_redsize_vgg16_loss_perceptual=args.prop_redsize_vgg16_loss_perceptual)
        model_trainer.create_list_metrics(list_type_metrics=args.list_type_metrics,
                                          is_mask_to_region_interest=args.is_mask_region_interest)
        model_trainer.finalise_model()

    if args.is_restart_model:
        model_restart_file = join_path_names(models_path, args.restart_file)
        print("Restart Model from file: \'%s\'..." % (model_restart_file))

        if args.is_restart_only_weights:
            if args.is_restart_weights_diffmodel:
                print("Load saved weights from a different model to restart model...")
                model_trainer.load_model_weights_diff_model(model_restart_file, type_load_model='UNet_noSkipConns')
            else:
                print("Load only saved weights to restart model...")
                model_trainer.load_model_only_weights(model_restart_file)
        else:
            print("Load full model (model weights and description, optimizer, loss and metrics) to restart model...")
            if TYPE_DNNLIB_USED == 'Keras':
                # CHECK WHETHER THIS IS NECESSARY
                model_trainer.create_loss(type_loss=args.type_loss,
                                          is_mask_to_region_interest=args.is_mask_region_interest,
                                          size_image_in=args.size_in_images,
                                          weight_combined_loss=args.weight_combined_loss,
                                          layers_vgg16_loss_perceptual=args.layers_vgg16_loss_perceptual,
                                          weights_vgg16_loss_perceptual=args.weights_vgg16_loss_perceptual,
                                          prop_redsize_vgg16_loss_perceptual=args.prop_redsize_vgg16_loss_perceptual)
                model_trainer.create_list_metrics(list_type_metrics=args.list_type_metrics,
                                                  is_mask_to_region_interest=args.is_mask_region_interest)
            if args.is_backward_compat:
                model_trainer.load_model_full_backward_compat(model_restart_file)
            else:
                model_trainer.load_model_full(model_restart_file)

    model_trainer.create_callbacks(models_path=models_path,
                                   is_restart_model=args.is_restart_model,
                                   is_validation_data=use_validation_data,
                                   freq_save_check_model=FREQ_SAVE_INTER_MODELS,
                                   freq_validate_model=FREQ_VALIDATE_MODEL)
    # model_trainer.summary_model()

    if (WRITE_OUT_DESC_MODEL_TEXT):
        out_descript_model_logfile = join_path_names(models_path, NAME_DESCRIPT_MODEL_LOGFILE)
        print("Write out descriptive model source model in text file: \'%s\'" % (out_descript_model_logfile))
        descmodel_text = model_trainer._networks.get_descmodel_sourcecode()
        fout = open(out_descript_model_logfile, 'w')
        fout.write(descmodel_text)
        fout.close()

    # *****************************************************

    # LOADING DATA
    print("\nLoading data...")
    print("-" * 30)

    size_out_image_model = model_trainer.get_size_output_image_model()

    if args.is_valid_convolutions:
        print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..."
              % (str(args.size_in_images), str(size_out_image_model)))

    print("Loading Training data...")
    training_data_loader = \
        get_train_imagedataloader_2images(list_train_images_files,
                                          list_train_labels_files,
                                          size_in_images=args.size_in_images,
                                          use_sliding_window_images=args.use_sliding_window_images,
                                          prop_overlap_slide_window=args.prop_overlap_sliding_window,
                                          use_transform_rigid_images=args.use_transform_rigid_images,
                                          use_transform_elastic_images=args.use_transform_elastic_images,
                                          use_random_window_images=args.use_random_window_images,
                                          num_random_patches_epoch=args.num_random_patches_epoch,
                                          is_nnet_validconvs=args.is_valid_convolutions,
                                          size_output_images=size_out_image_model,
                                          batch_size=args.batch_size,
                                          is_shuffle=IS_SHUFFLE_TRAINDATA,
                                          manual_seed=args.manual_seed_train,
                                          is_load_many_images_per_label=args.is_load_augmented_images_per_label,
                                          num_images_per_label=args.num_augmented_images_per_label,
                                          is_load_images_from_batches=args.is_train_images_slices)
    print("Loaded \'%s\' files. Total batches generated: %s..."
          % (len(list_train_images_files), len(training_data_loader)))

    if use_validation_data:
        print("\nLoading Validation data...")
        args.use_transform_rigid_images = \
            args.use_transform_rigid_images and USE_TRANSFORM_VALIDATION_DATA
        args.use_transform_elastic_images = \
            args.use_transform_elastic_images and USE_TRANSFORM_VALIDATION_DATA

        validation_data_loader = \
            get_train_imagedataloader_2images(list_valid_images_files,
                                              list_valid_labels_files,
                                              size_in_images=args.size_in_images,
                                              use_sliding_window_images=args.use_sliding_window_images,
                                              prop_overlap_slide_window=args.prop_overlap_sliding_window,
                                              use_transform_rigid_images=args.use_transform_rigid_images,
                                              use_transform_elastic_images=args.use_transform_elastic_images,
                                              use_random_window_images=args.use_random_window_images,
                                              num_random_patches_epoch=args.num_random_patches_epoch,
                                              is_nnet_validconvs=args.is_valid_convolutions,
                                              size_output_images=size_out_image_model,
                                              batch_size=args.batch_size,
                                              is_shuffle=IS_SHUFFLE_TRAINDATA,
                                              manual_seed=args.manual_seed_train,
                                              is_load_images_from_batches=args.is_train_images_slices)
        print("Loaded \'%s\' files. Total batches generated: %s..."
              % (len(list_valid_images_files), len(validation_data_loader)))
    else:
        validation_data_loader = None

    # *****************************************************

    # TRAINING MODEL
    print("\nTraining model...")
    print("-" * 30)

    if args.is_restart_model:
        if 'last' in basename(model_restart_file):
            loss_history_filename = join_path_names(models_path, NAME_LOSSHISTORY_FILE)
            restart_epoch = get_restart_epoch_from_loss_history_file(loss_history_filename)
        else:
            restart_epoch = get_restart_epoch_from_model_filename(model_restart_file)

        print('Restarting training from epoch \'%s\'...' % (restart_epoch))
        initial_epoch = restart_epoch
        args.num_epochs += initial_epoch
    else:
        initial_epoch = 0

    model_trainer.train(train_data_loader=training_data_loader,
                        valid_data_loader=validation_data_loader,
                        num_epochs=args.num_epochs,
                        max_steps_epoch=args.max_steps_epoch,
                        initial_epoch=initial_epoch,
                        is_shuffle_data=IS_SHUFFLE_TRAINDATA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--modelsdir', type=str, default='Models')
    parser.add_argument('--in_config_file', type=str, default=None)
    parser.add_argument('--size_in_images', type=str2tuple_int, default=SIZE_IN_IMAGES)
    parser.add_argument('--training_datadir', type=str, default=NAME_TRAININGDATA_RELPATH)
    parser.add_argument('--validation_datadir', type=str, default=NAME_VALIDATIONDATA_RELPATH)
    parser.add_argument('--max_train_images', type=int, default=MAX_TRAIN_IMAGES)
    parser.add_argument('--max_valid_images', type=int, default=MAX_VALID_IMAGES)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--max_steps_epoch', type=int, default=None)
    parser.add_argument('--type_network', type=str, default=TYPE_NETWORK)
    parser.add_argument('--net_num_levels', type=int, default=NET_NUM_LEVELS)
    parser.add_argument('--net_num_featmaps', type=int, default=NET_NUM_FEATMAPS)
    parser.add_argument('--type_optimizer', type=str, default=TYPE_OPTIMIZER)
    parser.add_argument('--learn_rate', type=float, default=LEARN_RATE)
    parser.add_argument('--type_loss', type=str, default=TYPE_LOSS)
    parser.add_argument('--weight_combined_loss', type=float, default=WEIGHT_COMBINED_LOSS)
    parser.add_argument('--list_type_metrics', type=str2list_str, default=LIST_TYPE_METRICS)
    parser.add_argument('--is_valid_convolutions', type=str2bool, default=IS_VALID_CONVOLUTIONS)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--use_sliding_window_images', type=str2bool, default=USE_SLIDING_WINDOW_IMAGES)
    parser.add_argument('--prop_overlap_sliding_window', type=str2tuple_float, default=PROP_OVERLAP_SLIDING_WINDOW)
    parser.add_argument('--use_random_window_images', type=str2bool, default=USE_RANDOM_WINDOW_IMAGES)
    parser.add_argument('--num_random_patches_epoch', type=str2int, default=NUM_RANDOM_PATCHES_EPOCH)
    parser.add_argument('--use_transform_rigid_images', type=str2bool, default=USE_TRANSFORM_RIGID_IMAGES)
    parser.add_argument('--use_transform_elastic_images', type=str2bool, default=USE_TRANSFORM_ELASTIC_IMAGES)
    parser.add_argument('--manual_seed_train', type=int, default=MANUAL_SEED_TRAIN)
    parser.add_argument('--name_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--is_restart_model', type=str2bool, default=False)
    parser.add_argument('--restart_file', type=str, default=NAME_SAVEDMODEL_LAST)
    parser.add_argument('--is_restart_only_weights', type=str2bool, default=IS_RESTART_ONLY_WEIGHTS)
    parser.add_argument('--is_backward_compat', type=str2bool, default=False)
    parser.add_argument('--is_load_augmented_images_per_label', type=str2bool, default=False)
    parser.add_argument('--num_augmented_images_per_label', type=int, default=None)
    parser.add_argument('--is_input_images_two_channels', type=str2bool, default=IS_MERGE_TWO_IMAGES_AS_CHANNELS)
    parser.add_argument('--is_train_network_2D', type=str2bool, default=False)
    parser.add_argument('--is_restart_weights_diffmodel', type=str2bool, default=False)
    parser.add_argument('--layers_vgg16_loss_perceptual', type=str2list_str, default=LAYERS_VGG16_LOSS_PERCEPTUAL)
    parser.add_argument('--weights_vgg16_loss_perceptual', type=str2list_float, default=WEIGHTS_VGG16_LOSS_PERCEPTUAL)
    parser.add_argument('--prop_redsize_vgg16_loss_perceptual', type=float, default=PROP_REDUSIZE_VGG16_LOSS_PERCEPTUAL)
    args = parser.parse_args()

    if args.is_restart_weights_diffmodel and not args.is_restart_only_weights:
        args.is_restart_only_weights = True

    if (args.is_restart_model and not args.is_restart_only_weights) and not args.in_config_file:
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
        args.max_train_images = int(input_args_file['max_train_images'])
        args.max_valid_images = int(input_args_file['max_valid_images'])
        args.batch_size = int(input_args_file['batch_size'])
        args.num_epochs = int(input_args_file['num_epochs'])
        args.max_steps_epoch = None     # CHECK THIS OUT!
        args.type_network = str(input_args_file['type_network'])
        args.net_num_levels = int(input_args_file['net_num_levels'])
        args.net_num_featmaps = int(input_args_file['net_num_featmaps'])
        args.type_optimizer = str(input_args_file['type_optimizer'])
        args.learn_rate = float(input_args_file['learn_rate'])
        args.type_loss = str(input_args_file['type_loss'])
        args.weight_combined_loss = float(input_args_file['weight_combined_loss'])
        args.list_type_metrics = str2list_str(input_args_file['list_type_metrics'])
        args.is_valid_convolutions = str2bool(input_args_file['is_valid_convolutions'])
        args.is_mask_region_interest = str2bool(input_args_file['is_mask_region_interest'])
        args.use_sliding_window_images = str2bool(input_args_file['use_sliding_window_images'])
        args.prop_overlap_sliding_window = str2tuple_float(input_args_file['prop_overlap_sliding_window'])
        args.use_random_window_images = str2bool(input_args_file['use_random_window_images'])
        args.num_random_patches_epoch = str2int(input_args_file['num_random_patches_epoch'])
        args.use_transform_rigid_images = str2bool(input_args_file['use_transform_rigid_images'])
        args.use_transform_elastic_images = str2bool(input_args_file['use_transform_elastic_images'])
        args.manual_seed_train = str2int(input_args_file['manual_seed_train'])
        args.name_reference_keys_file = str(input_args_file['name_reference_keys_file'])
        args.layers_vgg16_loss_perceptual = str2list_str(input_args_file['layers_vgg16_loss_perceptual'])
        args.weights_vgg16_loss_perceptual = str2list_float(input_args_file['weights_vgg16_loss_perceptual'])
        args.prop_redsize_vgg16_loss_perceptual = float(input_args_file['prop_redsize_vgg16_loss_perceptual'])

    if args.is_train_network_2D:
        print("Train 2D model, with images as slices from volume scans...")
        args.is_train_images_slices = True
        if len(args.size_in_images) != 2:
            message = "Wrong input \'size_in_images\': %s. To train 2D model, input size images must have 2 dims..." \
                      % (args.size_in_images)
            catch_error_exception(message)
    else:
        args.is_train_images_slices = False
        if len(args.size_in_images) != 3:
            message = "Wrong input \'size_in_images\': %s. To train 3D model, input size images must have 3 dims..." \
                      % (args.size_in_images)
            catch_error_exception(message)

    if args.is_load_augmented_images_per_label and not args.num_augmented_images_per_label:
        message = "When using the option to load augmented images from disc, " \
                  "need to input \'num_augmented_images_per_label\'..."
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in sorted(vars(args).items()):
        print("\'%s\' = %s" % (key, value))

    main(args)
