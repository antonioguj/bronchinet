#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

#from common.CPUGPUdevicesManager import *
from common.workdir_manager import *
from dataloaders.batchdatagenerator_manager import *
from dataloaders.loadimagedata_manager import *
if TYPE_DNNLIB_USED == 'Keras':
    from networks.keras.callbacks import *
    from networks.keras.metrics import *
    from Networks_Keras.Networks import *
elif TYPE_DNNLIB_USED == 'Pytorch':
    from networks.pytorch.metrics import *
    if ISTESTMODELSWITHGNN:
        from networks.NetworksGNNs import *
    else:
        from networks.Networks import *
    from networks.Trainers import *
from preprocessing.imagegenerator_manager import *
from collections import OrderedDict
import argparse


def write_train_valid_data_log_file(out_filename, list_data_files, dict_refer_keys, type_data):
    with open(out_filename, 'w') as fout:
        fout.write('Total of %s %s files\n' %(len(list_data_files), type_data))
        fout.write('---------------------------------\n')
        for in_file in list_data_files:
            fout.write('%s -> (%s)\n' % (basename(in_file), dict_refer_keys[basename_file_noext(in_file)]))
        #endfor



def main(args):
    # First thing, set session in the selected(s) devices: CPU or GPU
    #set_session_in_selected_device(use_GPU_device=True,
    #                               type_GPU_installed=TYPEGPUINSTALLED)

    # ---------- SETTINGS ----------
    nameInputImagesFiles = 'images_proc*.nii.gz'
    nameInputLabelsFiles = 'labels_proc*.nii.gz'
    nameInputExtraLabelsFiles = 'cenlines_proc*.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager    = GeneralDirManager(args.basedir)
    TrainingDataPath   = workDirsManager.get_pathdir_exist(args.traindatadir)
    InputReferKeysFile = workDirsManager.getNameExistBaseDataFile(args.nameInputReferKeysFile)

    listTrainImagesFiles = list_files_dir(TrainingDataPath, nameInputImagesFiles)[0:args.numMaxTrainImages]
    listTrainLabelsFiles = list_files_dir(TrainingDataPath, nameInputLabelsFiles)[0:args.numMaxTrainImages]
    in_dictReferenceKeys = read_dictionary(InputReferKeysFile)

    if args.restart_model:
        ModelsPath = workDirsManager.get_pathdir_exist(args.modelsdir)
    else:
        ModelsPath = workDirsManager.get_pathdir_update(args.modelsdir)

    if USEVALIDATIONDATA:
        ValidationDataPath   = workDirsManager.get_pathdir_exist(args.validdatadir)
        listValidImagesFiles = list_files_dir(ValidationDataPath, nameInputImagesFiles)[0:args.numMaxValidImages]
        listValidLabelsFiles = list_files_dir(ValidationDataPath, nameInputLabelsFiles)[0:args.numMaxValidImages]

        if not listValidImagesFiles or not listValidLabelsFiles:
            use_validation_data = False
            print("No validation data used for training the model...")
        else:
            use_validation_data = True
    else:
        use_validation_data = False



    # ----------------------------------------------
    # write out experiment parameters in config file
    out_configparams_file = join_path_names(ModelsPath, NAME_CONFIGPARAMS_FILE)
    if not is_exist_file(out_configparams_file):
        print("Write configuration parameters in file: \'%s\'..." % (out_configparams_file))
        dict_args = OrderedDict(sorted(vars(args).items()))
        save_dictionary_configparams(out_configparams_file, dict_args)

    # write out logs with the training and validation files files used
    out_logtraindata_file = join_path_names(ModelsPath, NAME_LOGTRAINDATA_FILE)
    if is_exist_file(out_logtraindata_file):
        out_logtraindata_file = update_filename(out_logtraindata_file)
    print("Write log for training data in file: \'%s\'..." % (out_logtraindata_file))
    write_train_valid_data_log_file(out_logtraindata_file, listTrainImagesFiles, in_dictReferenceKeys, 'training')

    if use_validation_data:
        out_logvaliddata_file = join_path_names(ModelsPath, NAME_LOGVALIDDATA_FILE)
        if is_exist_file(out_logvaliddata_file):
            out_logvaliddata_file = update_filename(out_logvaliddata_file)
        print("Write log for training data in file: \'%s\'..." % (out_logvaliddata_file))
        write_train_valid_data_log_file(out_logvaliddata_file, listValidImagesFiles, in_dictReferenceKeys, 'validation')
    # ----------------------------------------------



    # BUILDING MODEL
    # ----------------------------------------------
    print("-" * 30)
    print("Building model...")
    print("-" * 30)

    if args.restart_model:
        initial_epoch = args.restart_epoch
        args.num_epochs += initial_epoch
    else:
        initial_epoch = 0

    if TYPE_DNNLIB_USED == 'Keras':
        # ----------------------------------------------
        if (not args.restart_model) or\
        (args.restart_model and RESTART_ONLY_WEIGHTS):
            model_constructor = DICTAVAILMODELS3D(args.size_in_images,
                                                  num_levels=args.num_layers,
                                                  num_featmaps_in=args.num_featmaps_in,
                                                  isUse_valid_convols=args.isValidConvolutions)
                                                  # type_network=args.type_network,
                                                  # type_activate_hidden=args.type_activate_hidden,
                                                  # type_activate_output=args.type_activate_output,
                                                  # is_disable_convol_pooling_lastlayer=args.disable_convol_pooling_lastlayer,
                                                  # isuse_dropout=args.isUse_dropout,
                                                  # isuse_batchnormalize=args.isUse_batchnormalize)
            optimizer = get_optimizer_pytorch(args.optimizer, lr=args.learn_rate)
            loss_fun = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest).loss
            metrics =[DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest).get_renamed_compute() for imetrics in args.listmetrics]
            model = model_constructor.build_model()
            # compile model
            model.compile(optimizer=optimizer, loss=loss_fun, metrics=metrics)

            if args.isModel_halfPrecision:
                message = 'Networks implementation in Keras not available in Half Precision'
                catch_error_exception(message)

            if args.restart_model:
                print("Loading saved weights and restarting...")
                modelSavedPath = join_path_names(ModelsPath, 'model_last.hdf5')
                print("Restarting from file: \'%s\'..." %(modelSavedPath))
                model.load_weights(modelSavedPath)

        else: #args.restart_model and RESTART_ONLY_WEIGHTS:
            print("Loading full model: weights, optimizer, loss, metrics ... and restarting...")
            modelSavedPath = join_path_names(ModelsPath, 'model_last.hdf5')
            print("Restarting from file: \'%s\'..." %(modelSavedPath))

            loss_fun = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest).loss
            metrics  =[DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest).get_renamed_compute() for imetrics in args.listmetrics]
            custom_objects = dict(map(lambda fun: (fun.__name__, fun), [loss_fun] + metrics))
            # load and compile model
            model = NeuralNetwork.get_load_saved_model(modelSavedPath, custom_objects=custom_objects)

        # Callbacks:
        list_callbacks = []
        list_callbacks.append(RecordLossHistory(ModelsPath, NAME_LOSSHISTORY_FILE,
                                                [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest).get_renamed_compute() for imetrics in args.listmetrics]))
        filename = join_path_names(ModelsPath, 'model_e{epoch:02d}.hdf5')
        list_callbacks.append(callbacks.ModelCheckpoint(filename, monitor='loss', verbose=0))
        filename = join_path_names(ModelsPath, 'model_last.hdf5')
        list_callbacks.append(callbacks.ModelCheckpoint(filename, monitor='loss', verbose=0))
        # list_callbacks.append(callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='max'))

        size_output_modelnet = tuple(model_constructor.get_size_output()[1:])

        # output model summary
        model.summary()
        # ----------------------------------------------

    elif TYPE_DNNLIB_USED == 'Pytorch':
        # ----------------------------------------------
        if (not args.restart_model) or\
        (args.restart_model and RESTART_ONLY_WEIGHTS):
            if ISTESTMODELSWITHGNN:
                model_net = DICTAVAILMODELSGNNS(args.imodel,
                                                args.size_in_images,
                                                nfeat=args.num_featmaps_in,
                                                nlevel=args.num_layers,
                                                isUse_valid_convols=args.isValidConvolutions,
                                                isGNN_with_attention_lays=args.isGNNwithAttentionLays,
                                                source_dir_adjs=SOURCEDIR_ADJS)
            else:
                model_net = DICTAVAILMODELS3D(args.size_in_images,
                                              num_levels=args.num_layers,
                                              num_featmaps_in=args.num_featmaps_in,
                                              isUse_valid_convols=args.isValidConvolutions)
                                              # type_network=args.type_network,
                                              # type_activate_hidden=args.type_activate_hidden,
                                              # type_activate_output=args.type_activate_output,
                                              # is_disable_convol_pooling_lastlayer=args.disable_convol_pooling_lastlayer,
                                              # isuse_dropout=args.isUse_dropout,
                                              # isuse_batchnormalize=args.isUse_batchnormalize)

            optimizer = get_optimizer_pytorch(args.optimizer, model_net.parameters(), lr=args.learn_rate)
            loss_fun = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest)
            metrics_fun = [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest) for imetrics in args.listmetrics]

            trainer = Trainer(model_net, optimizer, loss_fun, metrics_fun,
                              model_halfprec=args.isModel_halfPrecision)

            if args.restart_model:
                print("Loading saved weights and restarting...")
                modelSavedPath = join_path_names(ModelsPath, 'model_last.pt')
                print("Restarting from file: \'%s\'..." %(modelSavedPath))
                trainer.load_model_only_weights(modelSavedPath)

        else: #args.restart_model and RESTART_ONLY_WEIGHTS:
            print("Loading full model: weights, optimizer, loss, metrics ... and restarting...")
            modelSavedPath = join_path_names(ModelsPath, 'model_last.pt')
            print("Restarting from file: \'%s\'..." %(modelSavedPath))

            dict_added_model_input_args = {'size_image': args.size_in_images,
                                           'num_levels': args.num_layers,
                                           'num_featmaps_in': args.num_featmaps_in,
                                           'isUse_valid_convols': args.isValidConvolutions}

            if ISTESTMODELSWITHGNN:
                dict_added_input_args_GNNmodel = {'isGNN_with_attention_lays': args.isGNNwithAttentionLays,
                                                  'source_dir_adjs': SOURCEDIR_ADJS}
                dict_added_model_input_args.update(dict_added_input_args_GNNmodel)


            metrics_fun = [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest) for imetrics in args.listmetrics]
            dict_added_other_input_args = {'metrics_fun': metrics_fun}

            if RESTART_FROMDIFFMODEL:
                print("Restarting weights of model \'%s\' from a different Unet model..." % (args.imodel))
                trainer = Trainer.load_model_from_diffmodel(modelSavedPath,
                                                            args.imodel,
                                                            dict_added_model_input_args=dict_added_model_input_args,
                                                            dict_added_other_input_args=dict_added_other_input_args)
            else:
                trainer = Trainer.load_model_full(modelSavedPath,
                                                  dict_added_model_input_args=dict_added_model_input_args,
                                                  dict_added_other_input_args=dict_added_other_input_args)

        trainer.setup_losshistory_filepath(ModelsPath, NAME_LOSSHISTORY_FILE,
                                           is_restart_file=args.restart_model)
        trainer.setup_validate_model(freq_validate_model=FREQVALIDATEMODEL)
        trainer.setup_savemodel_filepath(ModelsPath,
                                         type_save_models='full_model',
                                         freq_save_intermodels=FREQSAVEINTERMODELS)

        size_output_modelnet = tuple(trainer.model_net.get_size_output()[1:])

        # output model summary
        if not ISTESTMODELSWITHGNN and not args.isModel_halfPrecision:
            trainer.get_summary_model()
        # ----------------------------------------------

    if args.isValidConvolutions:
        print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..." % (str(args.size_in_images),
                                                                                               str(size_output_modelnet)))

    if (WRITEOUTDESCMODELTEXT):
        out_logdescmodel_file = join_path_names(ModelsPath, NAME_LOGDESCMODEL_FILE)
        # if not isExistfile(out_logdescmodel_file):
        print("Write out descriptive model source model in text file: \'%s\'" % (out_logdescmodel_file))
        descmodel_text = trainer.model_net.get_descmodel_sourcecode()
        fout = open(out_logdescmodel_file, 'w')
        fout.write(descmodel_text)
        fout.close()
    # ----------------------------------------------



    # LOADING DATA
    # ----------------------------------------------
    print("-" * 30)
    print("Loading data...")
    print("-" * 30)

    print("Load Training data...")
    if (args.slidingWindowImages or args.randomCropWindowImages or args.transformationRigidImages or args.transformElasticDeformImages):
        print("Generate Training images with Batch Generator of Training data...")

        (list_train_xData, list_train_yData) = LoadImageDataManager.load_2list_images(listTrainImagesFiles, listTrainLabelsFiles)

        train_batch_data_generator = get_batchdata_generator(args.size_in_images,
                                                             list_train_xData,
                                                             list_train_yData,
                                                             args.slidingWindowImages,
                                                             args.propOverlapSlidingWindow,
                                                             args.randomCropWindowImages,
                                                             args.numRandomImagesPerVolumeEpoch,
                                                             args.transformationRigidImages,
                                                             args.transformElasticDeformImages,
                                                             batch_size=args._batch_size,
                                                             is_output_nnet_validconvs=args.isValidConvolutions,
                                                             size_output_images=size_output_modelnet,
                                                             shuffle=SHUFFLETRAINDATA,
                                                             is_datagen_halfPrec=args.isModel_halfPrecision)
        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles),
                                                                          len(train_batch_data_generator)))
    else:
        (list_train_xData, list_train_yData) = LoadImageDataInBatchesManager(args.size_in_images).load_2list_images(listTrainImagesFiles,
                                                                                                                    listTrainLabelsFiles)
        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles),
                                                                          len(list_train_xData)))

    if use_validation_data:
        print("Load Validation data...")
        if (args.slidingWindowImages or args.randomCropWindowImages or args.transformationRigidImages or args.transformElasticDeformImages):
            print("Generate Validation images with Batch Generator of Validation data...")
            args.transformationRigidImages = args.transformationRigidImages and USETRANSFORMONVALIDATIONDATA
            args.transformElasticDeformImages = args.transformElasticDeformImages and USETRANSFORMONVALIDATIONDATA

            (list_valid_xData, list_valid_yData) = LoadImageDataManager.load_2list_images(listValidImagesFiles, listValidLabelsFiles)

            valid_batch_data_generator = get_batchdata_generator(args.size_in_images,
                                                                 list_valid_xData,
                                                                 list_valid_yData,
                                                                 args.slidingWindowImages,
                                                                 args.propOverlapSlidingWindow,
                                                                 args.randomCropWindowImages,
                                                                 args.numRandomImagesPerVolumeEpoch,
                                                                 args.transformationRigidImages,
                                                                 args.transformElasticDeformImages,
                                                                 batch_size=args._batch_size,
                                                                 is_output_nnet_validconvs=args.isValidConvolutions,
                                                                 size_output_images=size_output_modelnet,
                                                                 shuffle=SHUFFLETRAINDATA,
                                                                 is_datagen_halfPrec=args.isModel_halfPrecision)
            validation_data = valid_batch_data_generator
            print("Number volumes: %s. Total Data batches generated: %s..." %(len(listValidImagesFiles),
                                                                              len(valid_batch_data_generator)))
        else:
            (list_valid_xData, list_valid_yData) = LoadImageDataInBatchesManager(args.size_in_images).load_2list_images(listValidImagesFiles,
                                                                                                                        listValidLabelsFiles)
            validation_data = (list_valid_xData, list_valid_yData)
            print("Number volumes: %s. Total Data batches generated: %s..." % (len(listTrainImagesFiles),
                                                                               len(list_valid_xData)))
    else:
        validation_data = None
    # ----------------------------------------------



    # TRAINING MODEL
    # ----------------------------------------------
    print("-" * 30)
    print("Training model...")
    print("-" * 30)

    if TYPE_DNNLIB_USED == 'Keras':
        model.fit_generator(generator=train_batch_data_generator,
                            steps_per_epoch=args.max_steps_epoch,
                            epochs=args.num_epochs,
                            verbose=1,
                            callbacks=list_callbacks,
                            validation_data=validation_data,
                            shuffle=SHUFFLETRAINDATA,
                            initial_epoch=initial_epoch)

    elif TYPE_DNNLIB_USED == 'Pytorch':
        trainer.train(train_data_generator=train_batch_data_generator,
                      num_epochs=args.num_epochs,
                      max_steps_epoch=args.max_steps_epoch,
                      valid_data_generator=validation_data,
                      initial_epoch=initial_epoch)
    # ----------------------------------------------



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
    parser.add_argument('--imodel', type=str, default=IMODEL)
    parser.add_argument('--tailored_build_model', type=str2bool, default=TAILORED_BUILD_MODEL)
    parser.add_argument('--type_network', type=str, default=TYPE_NETWORK)
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--num_featmaps_in', type=int, default=NUM_FEATMAPS)
    parser.add_argument('--isUse_dropout', type=str2bool, default=ISUSE_DROPOUT)
    parser.add_argument('--isUse_batchnormalize', type=str2bool, default=ISUSE_BATCHNORMALIZE)
    parser.add_argument('--isModel_halfPrecision', type=str2bool, default=ISMODEL_HALFPRECISION)
    parser.add_argument('--type_activate_hidden', type=str, default=TYPE_ACTIVATE_HIDDEN)
    parser.add_argument('--type_activate_output', type=str, default=TYPE_ACTIVATE_OUTPUT)
    parser.add_argument('--type_padding_convol', type=str, default=TYPE_PADDING_CONVOL)
    parser.add_argument('--disable_convol_pooling_lastlayer', type=str2bool, default=DISABLE_CONVOL_POOLING_LASTLAYER)
    parser.add_argument('--optimizer', type=str, default=IOPTIMIZER)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--max_steps_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learn_rate', type=float, default=LEARN_RATE)
    parser.add_argument('--lossfun', type=str, default=ILOSSFUN)
    parser.add_argument('--listmetrics', type=str2list_string, default=LISTMETRICS)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--isValidConvolutions', type=str2bool, default=ISVALIDCONVOLUTIONS)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--propOverlapSlidingWindow', type=str2tuple_float, default=PROPOVERLAPSLIDINGWINDOW_Z_X_Y)
    parser.add_argument('--randomCropWindowImages', type=str2tuple_float, default=RANDOMCROPWINDOWIMAGES)
    parser.add_argument('--numRandomImagesPerVolumeEpoch', type=str2tuple_float, default=NUMRANDOMIMAGESPERVOLUMEEPOCH)
    parser.add_argument('--transformationRigidImages', type=str2bool, default=TRANSFORMATIONRIGIDIMAGES)
    parser.add_argument('--transformElasticDeformImages', type=str2bool, default=TRANSFORMELASTICDEFORMIMAGES)
    parser.add_argument('--restart_model', type=str2bool, default=RESTART_MODEL)
    parser.add_argument('--restart_epoch', type=int, default=RESTART_EPOCH)
    parser.add_argument('--isGNNwithAttentionLays', type=str2bool, default=ISGNNWITHATTENTIONLAYS)
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
        args.num_layers             = int(input_args_file['num_layers'])
        args.num_featmaps_in        = int(input_args_file['num_featmaps_in'])
        args.isUse_dropout          = str2bool(input_args_file['isUse_dropout'])
        args.isUse_batchnormalize   = str2bool(input_args_file['isUse_batchnormalize'])
        args.optimizer              = str(input_args_file['optimizer'])
        args.num_epochs             = int(input_args_file['num_epochs'])
        args.max_steps_epoch        = None # CHECK THIS OUT!
        args.batch_size             = int(input_args_file['batch_size'])
        args.learn_rate             = float(input_args_file['learn_rate'])
        args.lossfun                = str(input_args_file['lossfun'])
        args.listmetrics            = str2list_string(input_args_file['listmetrics'])
        args.listmetrics.remove('')
        args.masksToRegionInterest  = str2bool(input_args_file['masksToRegionInterest'])
        args.isValidConvolutions    = str2bool(input_args_file['isValidConvolutions'])
        args.slidingWindowImages    = str2bool(input_args_file['slidingWindowImages'])
        args.propOverlapSlidingWindow   = str2tuple_float(input_args_file['propOverlapSlidingWindow'])
        args.transformationRigidImages   = str2bool(input_args_file['transformationRigidImages'])
        args.transformElasticDeformImages= str2bool(input_args_file['transformElasticDeformImages'])
        args.isGNNwithAttentionLays = str2bool(input_args_file['isGNNwithAttentionLays'])

    print("Print input arguments...")
    for key, value in sorted(vars(args).items()):
        print("\'%s\' = %s" %(key, value))

    main(args)
