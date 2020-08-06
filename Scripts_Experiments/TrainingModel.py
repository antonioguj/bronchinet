
from common.constant import *
from common.functionutil import *
from common.workdirmanager import TrainDirManager
from dataloaders.dataloader_manager import get_train_imagedataloader_2images
from models.modeltrainer import ModelTrainer
from collections import OrderedDict
import argparse
if TYPE_DNNLIB_USED == 'Pytorch':
    NAME_SAVEDMODEL_LAST = NAME_SAVEDMODEL_LAST_TORCH
elif TYPE_DNNLIB_USED == 'Keras':
    NAME_SAVEDMODEL_LAST = NAME_SAVEDMODEL_LAST_KERAS


def write_train_valid_data_logfile(out_filename: str,
                                   list_data_files: List[str],
                                   dict_reference_keys: Dict[str, str],
                                   type_data: str
                                   ) -> None:
    if is_exist_file(out_filename):
        out_filename = update_filename(out_filename)
    print("Write log for train / valid data in file: \'%s\'..." % (out_filename))

    with open(out_filename, 'w') as fout:
        fout.write('Total of %s %s files\n' %(len(list_data_files), type_data))
        fout.write('---------------------------------\n')
        for in_file in list_data_files:
            fout.write('%s -> (%s)\n' % (basename(in_file), dict_reference_keys[basename_file_noext(in_file)]))

def get_restart_epoch_from_loss_history_file(loss_filename: str) -> int:
    data_file = np.genfromtxt(loss_filename, dtype=str, delimiter=' ')
    last_epoch_file = int(data_file[-1,0])      # retrieve the last epoch stored
    return last_epoch_file

def get_restart_epoch_from_model_filename(model_filename: str) -> int:
    pattern_search_epoch = 'e[0-9]+'
    epoch_filename = re.search(pattern_search_epoch, model_filename).group(0)
    epoch_filename = int(epoch_filename[1:])    # remove the leading 'e'
    return epoch_filename



def main(args):
    # ---------- SETTINGS ----------
    name_input_images_files = 'images_proc*.nii.gz'
    name_input_labels_files = 'labels_proc*.nii.gz'
    name_input_extra_labels_files = 'cenlines_proc*.nii.gz'
    # ---------- SETTINGS ----------


    workdir_manager          = TrainDirManager(args.basedir)
    training_data_path       = workdir_manager.get_pathdir_exist(args.traindatadir)
    input_reference_keys_file= workdir_manager.get_datafile_exist(args.nameInputReferKeysFile)

    list_train_images_files = list_files_dir(training_data_path, name_input_images_files)[0:args.numMaxTrainImages]
    list_train_labels_files = list_files_dir(training_data_path, name_input_labels_files)[0:args.numMaxTrainImages]
    dict_reference_keys     = read_dictionary(input_reference_keys_file)

    if args.is_restart_model:
        models_path = workdir_manager.get_pathdir_exist(args.modelsdir)
    else:
        models_path = workdir_manager.get_pathdir_update(args.modelsdir)

    if USEVALIDATIONDATA:
        validation_data_path    = workdir_manager.get_pathdir_exist(args.validdatadir)
        list_valid_images_files = list_files_dir(validation_data_path, name_input_images_files)[0:args.numMaxValidImages]
        list_valid_labels_files = list_files_dir(validation_data_path, name_input_labels_files)[0:args.numMaxValidImages]

        if not list_valid_images_files or not list_valid_labels_files:
            use_validation_data = False
            print("No validation data used for Training the model...")
        else:
            use_validation_data = True
    else:
        use_validation_data = False


    # write out experiment parameters in config file
    out_configparams_file = join_path_names(models_path, NAME_CONFIGPARAMS_FILE)
    if not is_exist_file(out_configparams_file):
        print("Write configuration parameters in file: \'%s\'..." % (out_configparams_file))
        dict_args = OrderedDict(sorted(vars(args).items()))
        save_dictionary_configparams(out_configparams_file, dict_args)

    # write out logs with the training and validation files files used
    out_traindata_logfile = join_path_names(models_path, NAME_TRAINDATA_LOGFILE)
    write_train_valid_data_logfile(out_traindata_logfile, list_train_images_files, dict_reference_keys, 'training')

    if use_validation_data:
        out_validdata_logfile = join_path_names(models_path, NAME_VALIDDATA_LOGFILE)
        write_train_valid_data_logfile(out_validdata_logfile, list_valid_images_files, dict_reference_keys, 'validation')


    # BUILDING MODEL
    print("\nBuilding model...")
    print("-" * 30)

    model_trainer = ModelTrainer()

    if not args.is_restart_model or (args.is_restart_model and IS_RESTART_ONLY_WEIGHTS):
        model_trainer.create_network(type_network=args.type_network,
                                     size_image_in=args.size_in_images,
                                     num_levels=args.num_levels,
                                     num_featmaps_in=args.num_featmaps,
                                     num_channels_in=1,
                                     num_classes_out=1,
                                     is_use_valid_convols=args.is_valid_convolutions)
        model_trainer.create_optimizer(type_optimizer=args.type_optimizer,
                                       learn_rate=args.learn_rate)
        model_trainer.create_loss(type_loss=args.type_loss,
                                  is_mask_to_region_interest=args.is_masks_to_region_interest)
        model_trainer.create_list_metrics(list_type_metrics=args.list_type_metrics,
                                          is_mask_to_region_interest=args.is_masks_to_region_interest)
        model_trainer.compile_model()

        model_trainer.create_callbacks(models_path=models_path,
                                       freq_save_check_model=FREQ_SAVE_INTER_MODELS,
                                       freq_validate_model=FREQ_VALIDATE_MODEL)

    if args.is_restart_model:
        model_restart_file = join_path_names(models_path, args.restart_file)
        print("Restart Model from file: \'%s\'..." % (model_restart_file))

        if IS_RESTART_ONLY_WEIGHTS:
            print("Load only saved weights to restart model...")
            model_trainer.load_model_only_weights(model_restart_file)
        else:
            print("Load full model: weights, model desc, optimizer, ... to restart model...")
            if TYPE_DNNLIB_USED == 'Keras':
                # CHECK WHETHER THIS IS NECESSARY
                model_trainer.create_loss(type_loss=args.type_loss,
                                          is_mask_to_region_interest=args.is_masks_to_region_interest)
                model_trainer.create_list_metrics(list_type_metrics=args.list_type_metrics,
                                                  is_mask_to_region_interest=args.is_masks_to_region_interest)
            model_trainer.load_model_full(model_restart_file)

    model_trainer.summary_model()

    if (WRITE_OUT_DESC_MODEL_TEXT):
        out_descmodel_logfile = join_path_names(models_path, NAME_DESCMODEL_LOGFILE)
        print("Write out descriptive model source model in text file: \'%s\'" % (out_descmodel_logfile))
        descmodel_text = model_trainer._networks.get_descmodel_sourcecode()
        fout = open(out_descmodel_logfile, 'w')
        fout.write(descmodel_text)
        fout.close()


    # LOADING DATA
    print("\nLoading data...")
    print("-" * 30)

    size_out_image_model = model_trainer.get_size_output_image_model()

    if args.is_valid_convolutions:
        print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..." % (str(args.size_in_images),
                                                                                               str(size_out_image_model)))
    print("Load Validation data...")
    training_data_loader = get_train_imagedataloader_2images(list_train_images_files,
                                                             list_train_labels_files,
                                                             args.size_in_images,
                                                             args.slidingWindowImages,
                                                             args.propOverlapSlidingWindow,
                                                             args.randomCropWindowImages,
                                                             args.numRandomImagesPerVolumeEpoch,
                                                             args.transformationRigidImages,
                                                             args.transformElasticDeformImages,
                                                             is_output_nnet_validconvs=args.is_valid_convolutions,
                                                             size_output_images=size_out_image_model,
                                                             batch_size=args._batch_size,
                                                             shuffle=SHUFFLETRAINDATA)
    print("Loaded \'%s\' files. Total batches generated: %s..." % (len(list_train_images_files),
                                                                   len(training_data_loader)))

    if use_validation_data:
        print("Load Validation data...")
        args.transformationRigidImages = args.transformationRigidImages and USE_TRANSFORM_VALIDATION_DATA
        args.transformElasticDeformImages = args.transformElasticDeformImages and USE_TRANSFORM_VALIDATION_DATA

        validation_data_loader = get_train_imagedataloader_2images(list_valid_images_files,
                                                                   list_valid_labels_files,
                                                                   args.size_in_images,
                                                                   args.slidingWindowImages,
                                                                   args.propOverlapSlidingWindow,
                                                                   args.randomCropWindowImages,
                                                                   args.numRandomImagesPerVolumeEpoch,
                                                                   args.transformationRigidImages,
                                                                   args.transformElasticDeformImages,
                                                                   is_output_nnet_validconvs=args.is_valid_convolutions,
                                                                   size_output_images=size_out_image_model,
                                                                   batch_size=args._batch_size,
                                                                   shuffle=SHUFFLETRAINDATA)
        print("Loaded \'%s\' files. Total batches generated: %s..." % (len(list_valid_images_files),
                                                                       len(validation_data_loader)))
    else:
        validation_data_loader = None


    # TRAINING MODEL
    print("\nTraining model...")
    print("-" * 30)

    if args.is_restart_model:
        if 'last' in basename(model_restart_file):
            loss_history_filename = join_path_names(models_path, NAME_LOSSHISTORY_FILE)
            restart_epoch = get_restart_epoch_from_loss_history_file(loss_history_filename)
        else:
            restart_epoch = get_restart_epoch_from_model_filename(model_restart_file)

        initial_epoch = restart_epoch
        args._num_epochs += initial_epoch
    else:
        initial_epoch = 0

    model_trainer.train(train_data_loader=training_data_loader,
                        valid_data_loader=validation_data_loader,
                        num_epochs=args.num_epochs,
                        max_steps_epoch=args.max_steps_epoch,
                        initial_epoch=initial_epoch)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--modelsdir', type=str, default='Models')
    parser.add_argument('--cfgfromfile', type=str, default=None)
    parser.add_argument('--size_in_images', type=str2tuple_int, default=IMAGES_DIMS_Z_X_Y)
    parser.add_argument('--traindatadir', type=str, default=NAME_TRAININGDATA_RELPATH)
    parser.add_argument('--validdatadir', type=str, default=NAME_VALIDATIONDATA_RELPATH)
    parser.add_argument('--nameInputReferKeysFile', type=str, default=NAME_REFERKEYSPROCIMAGE_FILE)
    parser.add_argument('--numMaxTrainImages', type=int, default=NUMMAXTRAINIMAGES)
    parser.add_argument('--numMaxValidImages', type=int, default=NUMMAXVALIDIMAGES)
    parser.add_argument('--type_network', type=str, default=TYPE_NETWORK)
    parser.add_argument('--tailored_build_model', type=str2bool, default=TAILORED_BUILD_MODEL)
    parser.add_argument('--num_levels', type=int, default=NUM_LEVELS)
    parser.add_argument('--num_featmaps_in', type=int, default=NUM_FEATMAPS)
    parser.add_argument('--isUse_dropout', type=str2bool, default=ISUSE_DROPOUT)
    parser.add_argument('--isUse_batchnormalize', type=str2bool, default=ISUSE_BATCHNORMALIZE)
    parser.add_argument('--isModel_halfPrecision', type=str2bool, default=ISMODEL_HALFPRECISION)
    parser.add_argument('--type_activate_hidden', type=str, default=TYPE_ACTIVATE_HIDDEN)
    parser.add_argument('--type_activate_output', type=str, default=TYPE_ACTIVATE_OUTPUT)
    parser.add_argument('--type_padding_convol', type=str, default=TYPE_PADDING_CONVOL)
    parser.add_argument('--disable_convol_pooling_lastlayer', type=str2bool, default=DISABLE_CONVOL_POOLING_LASTLAYER)
    parser.add_argument('--type_optimizer', type=str, default=TYPE_OPTIMIZER)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--max_steps_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learn_rate', type=float, default=LEARN_RATE)
    parser.add_argument('--type_loss', type=str, default=TYPE_LOSS)
    parser.add_argument('--list_type_metrics', type=str2list_string, default=LIST_TYPE_METRICS)
    parser.add_argument('--is_masks_to_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--is_valid_convolutions', type=str2bool, default=IS_VALID_CONVOLUTIONS)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--propOverlapSlidingWindow', type=str2tuple_float, default=PROPOVERLAPSLIDINGWINDOW_Z_X_Y)
    parser.add_argument('--randomCropWindowImages', type=str2tuple_float, default=RANDOMCROPWINDOWIMAGES)
    parser.add_argument('--numRandomImagesPerVolumeEpoch', type=str2tuple_float, default=NUMRANDOMIMAGESPERVOLUMEEPOCH)
    parser.add_argument('--transformationRigidImages', type=str2bool, default=TRANSFORMATIONRIGIDIMAGES)
    parser.add_argument('--transformElasticDeformImages', type=str2bool, default=TRANSFORMELASTICDEFORMIMAGES)
    parser.add_argument('--is_restart_model', type=str2bool, default=IS_RESTART_MODEL)
    parser.add_argument('--restart_file', type=str, default=NAME_SAVEDMODEL_LAST)
    args = parser.parse_args()

    if args.cfgfromfile:
        if not is_exist_file(args.cfgfromfile):
            message = "Config params file not found: \'%s\'..." %(args.cfgfromfile)
            catch_error_exception(message)
        else:
            input_args_file = read_dictionary_configparams(args.cfgfromfile)
        print("Set up experiments with parameters from file: \'%s\'" %(args.cfgfromfile))
        args.basedir                = str(input_args_file['basedir'])
        args.size_in_images         = str2tuple_int(input_args_file['size_in_images'])
        args.traindatadir           = str(input_args_file['traindatadir'])
        args.validdatadir           = str(input_args_file['validdatadir'])
        args.numMaxTrainImages      = int(input_args_file['numMaxTrainImages'])
        args.numMaxValidImages      = int(input_args_file['numMaxValidImages'])
        args.imodel                 = str(input_args_file['imodel'])
        args.num_levels             = int(input_args_file['num_layers'])
        args.num_featmaps_in        = int(input_args_file['num_featmaps_in'])
        args.isUse_dropout          = str2bool(input_args_file['isUse_dropout'])
        args.isUse_batchnormalize   = str2bool(input_args_file['isUse_batchnormalize'])
        args.type_optimizer              = str(input_args_file['type_optimizer'])
        args.num_epochs             = int(input_args_file['num_epochs'])
        args.max_steps_epoch        = None # CHECK THIS OUT!
        args.batch_size             = int(input_args_file['batch_size'])
        args.learn_rate             = float(input_args_file['learn_rate'])
        args.type_loss                = str(input_args_file['type_loss'])
        args.list_type_metrics            = str2list_string(input_args_file['listmetrics'])
        args.list_type_metrics.remove('')
        args.is_masks_to_region_interest  = str2bool(input_args_file['is_masks_to_region_interest'])
        args.is_valid_convolutions    = str2bool(input_args_file['is_valid_convolutions'])
        args.slidingWindowImages    = str2bool(input_args_file['slidingWindowImages'])
        args.propOverlapSlidingWindow   = str2tuple_float(input_args_file['propOverlapSlidingWindow'])
        args.transformationRigidImages   = str2bool(input_args_file['transformationRigidImages'])
        args.transformElasticDeformImages= str2bool(input_args_file['transformElasticDeformImages'])
        args.isGNNwithAttentionLays = str2bool(input_args_file['isGNNwithAttentionLays'])

    print("Print input arguments...")
    for key, value in sorted(vars(args).items()):
        print("\'%s\' = %s" %(key, value))

    main(args)