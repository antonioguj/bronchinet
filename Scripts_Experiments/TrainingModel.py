#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.CPUGPUdevicesManager import *
from Common.WorkDirsManager import *
from DataLoaders.LoadDataManager import *
if TYPE_DNNLIBRARY_USED == 'Keras':
    from DataLoaders.BatchDataGenerator_Keras import *
    from Networks_Keras.Callbacks import *
    from Networks_Keras.Metrics import *
    from Networks_Keras.Networks import *
    from Networks_Keras.Optimizers import *
elif TYPE_DNNLIBRARY_USED == 'Pytorch':
    from DataLoaders.BatchDataGenerator_Pytorch import *
    from Networks_Pytorch.Callbacks import *
    from Networks_Pytorch.Metrics import *
    if ISTESTMODELSWITHGNN:
        from Networks_Pytorch.NetworksGNNs import *
    else:
        from Networks_Pytorch.Networks import *
    from Networks_Pytorch.Optimizers import *
    from Networks_Pytorch.Trainers import *
from Preprocessing.ImageGeneratorManager import *
from collections import OrderedDict
import argparse



def main(args):
    # First thing, set session in the selected(s) devices: CPU or GPU
    set_session_in_selected_device(use_GPU_device=True,
                                   type_GPU_installed=TYPEGPUINSTALLED)

    # ---------- SETTINGS ----------
    nameModelsRelPath  = args.modelsdir
    nameImagesFiles    = 'images*' + getFileExtension(FORMATTRAINDATA)
    nameLabelsFiles    = 'labels*' + getFileExtension(FORMATTRAINDATA)
    cfgparams_filename = 'cfgparams.txt'
    descmodel_filename = 'descmodel.txt'
    # ---------- SETTINGS ----------


    workDirsManager  = WorkDirsManager(args.basedir)
    TrainingDataPath = workDirsManager.getNameExistPath(args.traindatadir)
    if args.restart_model:
        ModelsPath = workDirsManager.getNameExistPath(nameModelsRelPath)
    else:
        ModelsPath = workDirsManager.getNameUpdatePath(nameModelsRelPath)

    listTrainImagesFiles = findFilesDirAndCheck(TrainingDataPath, nameImagesFiles)[0:args.numMaxTrainImages]
    listTrainLabelsFiles = findFilesDirAndCheck(TrainingDataPath, nameLabelsFiles)[0:args.numMaxTrainImages]

    if USEVALIDATIONDATA:
        ValidationDataPath = workDirsManager.getNameExistPath(args.validdatadir)

        listValidImagesFiles = findFilesDir(ValidationDataPath, nameImagesFiles)[0:args.numMaxValidImages]
        listValidLabelsFiles = findFilesDir(ValidationDataPath, nameLabelsFiles)[0:args.numMaxValidImages]

        if not listValidImagesFiles or not listValidLabelsFiles:
            use_validation_data = False
            message = "No validation data used for training the model..."
            CatchWarningException(message)
        else:
            use_validation_data = True
    else:
        use_validation_data = False

    # write out experiment parameters in config file
    cfgparams_filename = joinpathnames(ModelsPath, cfgparams_filename)
    if not isExistfile(cfgparams_filename):
        print("Write out config parameters in file: \'%s\'" % (cfgparams_filename))
        dict_args = OrderedDict(sorted(vars(args).iteritems()))
        saveDictionary_configParams(cfgparams_filename, dict_args)


    
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

    if TYPE_DNNLIBRARY_USED == 'Keras':
        # ----------------------------------------------
        if (not args.restart_model) or\
        (args.restart_model and RESTART_ONLY_WEIGHTS):
            model_constructor = DICTAVAILMODELS3D(args.size_in_images,
                                                  num_layers=args.num_layers,
                                                  num_featmaps_in=args.num_featmaps_in,
                                                  tailored_build_model=args.tailored_build_model,
                                                  type_network=args.type_network,
                                                  type_activate_hidden=args.type_activate_hidden,
                                                  type_activate_output=args.type_activate_output,
                                                  type_padding_convol=args.type_padding_convol,
                                                  is_disable_convol_pooling_lastlayer=args.disable_convol_pooling_lastlayer,
                                                  isuse_dropout=args.isUse_dropout,
                                                  isuse_batchnormalize=args.isUse_batchnormalize)
            optimizer = DICTAVAILOPTIMIZERS(args.optimizer, lr=args.learn_rate)
            loss_fun = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest).loss
            metrics =[DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest).get_renamed_compute() for imetrics in args.listmetrics]
            model = model_constructor.get_model()
            # compile model
            model.compile(optimizer=optimizer, loss=loss_fun, metrics=metrics)

            if args.restart_model:
                print("Loading saved weights and restarting...")
                modelSavedPath = joinpathnames(ModelsPath, 'model_last.hdf5')
                print("Restarting from file: \'%s\'..." %(modelSavedPath))
                model.load_weights(modelSavedPath)

        else: #args.restart_model and RESTART_ONLY_WEIGHTS:
            print("Loading full model: weights, optimizer, loss, metrics ... and restarting...")
            modelSavedPath = joinpathnames(ModelsPath, 'model_last.hdf5')
            print("Restarting from file: \'%s\'..." %(modelSavedPath))

            loss_fun = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest).loss
            metrics  =[DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest).get_renamed_compute() for imetrics in args.listmetrics]
            custom_objects = dict(map(lambda fun: (fun.__name__, fun), [loss_fun] + metrics))
            # load and compile model
            model = NeuralNetwork.get_load_saved_model(modelSavedPath, custom_objects=custom_objects)

        # Callbacks:
        list_callbacks = []
        list_callbacks.append(RecordLossHistory(ModelsPath, [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest).get_renamed_compute() for imetrics in args.listmetrics]))
        filename = joinpathnames(ModelsPath, 'model_e{epoch:02d}.hdf5')
        list_callbacks.append(callbacks.ModelCheckpoint(filename, monitor='loss', verbose=0))
        filename = joinpathnames(ModelsPath, 'model_last.hdf5')
        list_callbacks.append(callbacks.ModelCheckpoint(filename, monitor='loss', verbose=0))
        # list_callbacks.append(callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='max'))

        # size_output_modelnet = tuple(trainer.model_net.get_size_output()[1:])
        size_output_modelnet = args.size_in_images  # IMPLEMENT HERE HOW TO COMPUTE SIZE OF OUTPUT MODEL
        if args.isValidConvolutions:
            message = "CODE WITH KERAS NOT IMPLEMENTED FOR VALID CONVOLUTIONS..."
            CatchErrorException(message)

        # output model summary
        model.summary()
        # ----------------------------------------------

    elif TYPE_DNNLIBRARY_USED == 'Pytorch':
        # ----------------------------------------------
        if (not args.restart_model) or\
        (args.restart_model and RESTART_ONLY_WEIGHTS):
            if ISTESTMODELSWITHGNN:
                model_net = DICTAVAILMODELSGNNS(args.imodel,
                                                args.size_in_images,
                                                nfeat=args.num_featmaps_in,
                                                nlevel=args.num_layers,
                                                isUse_valid_convs=args.isValidConvolutions,
                                                isGNN_with_attention_lays=args.isGNNwithAttentionLays,
                                                source_dir_adjs=SOURCEDIR_ADJS)
            else:
                model_net = DICTAVAILMODELS3D(args.size_in_images,
                                              num_layers=args.num_layers,
                                              num_featmaps_in=args.num_featmaps_in,
                                              tailored_build_model=args.tailored_build_model,
                                              type_network=args.type_network,
                                              type_activate_hidden=args.type_activate_hidden,
                                              type_activate_output=args.type_activate_output,
                                              type_padding_convol=args.type_padding_convol,
                                              is_disable_convol_pooling_lastlayer=args.disable_convol_pooling_lastlayer,
                                              isuse_dropout=args.isUse_dropout,
                                              isuse_batchnormalize=args.isUse_batchnormalize)

            optimizer = DICTAVAILOPTIMIZERS(args.optimizer, model_net.parameters(), lr=args.learn_rate)
            loss_fun = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest)
            metrics_fun = [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest) for imetrics in LISTMETRICS]

            trainer = Trainer(model_net, optimizer, loss_fun, metrics_fun)

            if args.restart_model:
                print("Loading saved weights and restarting...")
                modelSavedPath = joinpathnames(ModelsPath, 'model_last.pt')
                print("Restarting from file: \'%s\'..." %(modelSavedPath))
                trainer.load_model_only_weights(modelSavedPath)

        else: #args.restart_model and RESTART_ONLY_WEIGHTS:
            print("Loading full model: weights, optimizer, loss, metrics ... and restarting...")
            modelSavedPath = joinpathnames(ModelsPath, 'model_last.pt')
            print("Restarting from file: \'%s\'..." %(modelSavedPath))

            if ISTESTMODELSWITHGNN:
                dict_added_model_input_args = {'isUse_valid_convs': args.isValidConvolutions,
                                               'isGNN_with_attention_lays': args.isGNNwithAttentionLays,
                                               'source_dir_adjs': SOURCEDIR_ADJS}
            else:
                dict_added_model_input_args = {}

            metrics_fun = [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest) for imetrics in LISTMETRICS]
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

        trainer.setup_losshistory_filepath(ModelsPath,
                                           isexists_lossfile=args.restart_model)
        trainer.setup_validate_model(freq_validate_model=FREQVALIDATEMODEL)
        trainer.setup_savemodel_filepath(ModelsPath,
                                         type_save_models='full_model',
                                         freq_save_intermodels=FREQSAVEINTERMODELS)

        # size_output_modelnet = tuple(trainer.model_net.get_size_output()[1:])
        # if args.isValidConvolutions:
        #     print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..." % (str(args.size_in_images),
        #                                                                                            str(size_output_modelnet)))
        size_output_modelnet = args.size_in_images  # IMPLEMENT HERE HOW TO COMPUTE SIZE OF OUTPUT MODEL
        if args.isValidConvolutions:
            message = "CODE WITH KERAS NOT IMPLEMENTED FOR VALID CONVOLUTIONS..."
            CatchErrorException(message)

        # output model summary
        trainer.get_summary_model()
        # ----------------------------------------------

    if (WRITEOUTDESCMODELTEXT):
        descmodel_filename = joinpathnames(ModelsPath, descmodel_filename)
        # if not isExistfile(descmodel_filename):
        print("Write out descriptive model source model in text file: \'%s\'" % (descmodel_filename))
        descmodel_txt = trainer.model_net.get_descmodel_sourcecode()
        fout = open(descmodel_filename, 'w')
        fout.write(descmodel_txt)
        fout.close()
    # ----------------------------------------------



    # LOADING DATA
    # ----------------------------------------------
    print("-" * 30)
    print("Loading data...")
    print("-" * 30)

    print("Load Training data...")
    if (args.slidingWindowImages or args.transformationImages or args.elasticDeformationImages):
        print("Generate Training images with Batch Generator of Training data...")
        
        (list_train_xData, list_train_yData) = LoadDataManager.loadData_ListFiles(listTrainImagesFiles,
                                                                                  listTrainLabelsFiles)
        train_images_generator = getImagesDataGenerator3D(args.size_in_images,
                                                          args.slidingWindowImages,
                                                          args.slidewin_propOverlap,
                                                          args.transformationImages,
                                                          args.elasticDeformationImages)
        train_batch_data_generator = TrainingBatchDataGenerator(args.size_in_images,
                                                                list_train_xData,
                                                                list_train_yData,
                                                                train_images_generator,
                                                                batch_size=args.batch_size,
                                                                isUse_valid_convs=args.isValidConvolutions,
                                                                size_output_model=size_output_modelnet,
                                                                shuffle=SHUFFLETRAINDATA)
        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles),
                                                                          len(train_batch_data_generator)))
    else:
        (list_train_xData, list_train_yData) = LoadDataManagerInBatches(args.size_in_images).loadData_ListFiles(listTrainImagesFiles,
                                                                                                                listTrainLabelsFiles)
        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles),
                                                                          len(list_train_xData)))

    if use_validation_data:
        print("Load Validation data...")
        if (args.slidingWindowImages or args.transformationImages or args.elasticDeformationImages):
            print("Generate Validation images with Batch Generator of Validation data...")
            args.transformationImages = args.transformationImages and USETRANSFORMONVALIDATIONDATA
            args.elasticDeformationImages = args.elasticDeformationImages and USETRANSFORMONVALIDATIONDATA
            
            (list_valid_xData, list_valid_yData) = LoadDataManager.loadData_ListFiles(listValidImagesFiles,
                                                                                      listValidLabelsFiles)
            valid_images_generator = getImagesDataGenerator3D(args.size_in_images,
                                                              args.slidingWindowImages,
                                                              args.slidewin_propOverlap,
                                                              args.transformationImages,
                                                              args.elasticDeformationImages)
            valid_batch_data_generator = TrainingBatchDataGenerator(args.size_in_images,
                                                                    list_valid_xData,
                                                                    list_valid_yData,
                                                                    valid_images_generator,
                                                                    batch_size=args.batch_size,
                                                                    isUse_valid_convs=args.isValidConvolutions,
                                                                    size_output_model=size_output_modelnet,
                                                                    shuffle=SHUFFLETRAINDATA)
            validation_data = valid_batch_data_generator
            print("Number volumes: %s. Total Data batches generated: %s..." %(len(listValidImagesFiles),
                                                                              len(valid_batch_data_generator)))
        else:
            (list_valid_xData, list_valid_yData) = LoadDataManagerInBatches(args.size_in_images).loadData_ListFiles(listValidImagesFiles,
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

    if TYPE_DNNLIBRARY_USED == 'Keras':
        if (args.slidingWindowImages or
            args.transformationImages):
            model.fit_generator(train_batch_data_generator,
                                nb_epoch=args.num_epochs,
                                steps_per_epoch=args.max_steps_epoch,
                                verbose=1,
                                callbacks=list_callbacks,
                                validation_data=validation_data,
                                shuffle=SHUFFLETRAINDATA,
                                initial_epoch=initial_epoch)
        else:
            model.fit(list_train_xData,
                      list_train_yData,
                      batch_size=args.batch_size,
                      epochs=args.num_epochs,
                      steps_per_epoch=args.max_steps_epoch,
                      verbose=1,
                      callbacks=list_callbacks,
                      validation_data=validation_data,
                      shuffle=SHUFFLETRAINDATA,
                      initial_epoch=initial_epoch)

    elif TYPE_DNNLIBRARY_USED == 'Pytorch':
        trainer.train(train_batch_data_generator,
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
    parser.add_argument('--size_in_images', type=str2tupleint, default=IMAGES_DIMS_Z_X_Y)
    parser.add_argument('--traindatadir', type=str, default='TrainingData')
    parser.add_argument('--validdatadir', type=str, default='ValidationData')
    parser.add_argument('--numMaxTrainImages', type=int, default=NUMMAXTRAINIMAGES)
    parser.add_argument('--numMaxValidImages', type=int, default=NUMMAXVALIDIMAGES)
    parser.add_argument('--imodel', type=str, default=IMODEL)
    parser.add_argument('--tailored_build_model', type=str2bool, default=TAILORED_BUILD_MODEL)
    parser.add_argument('--type_network', type=str, default=TYPE_NETWORK)
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--num_featmaps_in', type=int, default=NUM_FEATMAPS)
    parser.add_argument('--isUse_dropout', type=str2bool, default=ISUSE_DROPOUT)
    parser.add_argument('--isUse_batchnormalize', type=str2bool, default=ISUSE_BATCHNORMALIZE)
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
    parser.add_argument('--listmetrics', type=parseListarg, default=LISTMETRICS)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--isValidConvolutions', type=str2bool, default=ISVALIDCONVOLUTIONS)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--slidewin_propOverlap', type=str2tuplefloat, default=SLIDEWIN_PROPOVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=ELASTICDEFORMATIONIMAGES)
    parser.add_argument('--restart_model', type=str2bool, default=RESTART_MODEL)
    parser.add_argument('--restart_epoch', type=int, default=RESTART_EPOCH)
    parser.add_argument('--isGNNwithAttentionLays', type=str2bool, default=ISGNNWITHATTENTIONLAYS)
    args = parser.parse_args()

    if args.cfgfromfile:
        if not isExistfile(args.cfgfromfile):
            message = "Config params file not found: \'%s\'..." %(args.cfgfromfile)
            CatchErrorException(message)
        else:
            input_args_file = readDictionary_configParams(args.cfgfromfile)
        print("Set up experiments with parameters from file: \'%s\'" %(args.cfgfromfile))
        args.basedir                = str(input_args_file['basedir'])
        args.size_in_images         = str2tupleint(input_args_file['size_in_images'])
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
        args.listmetrics            = str(input_args_file['listmetrics'])
        args.masksToRegionInterest  = str2bool(input_args_file['masksToRegionInterest'])
        args.isValidConvolutions    = str2bool(input_args_file['isValidConvolutions'])
        args.slidingWindowImages    = str2bool(input_args_file['slidingWindowImages'])
        args.slidewin_propOverlap   = str2tuplefloat(input_args_file['slidewin_propOverlap'])
        args.transformationImages   = str2bool(input_args_file['transformationImages'])
        args.elasticDeformationImages= str2bool(input_args_file['elasticDeformationImages'])
        args.isGNNwithAttentionLays = str2bool(input_args_file['isGNNwithAttentionLays'])

    print("Print input arguments...")
    for key, value in sorted(vars(args).iteritems()):
        print("\'%s\' = %s" %(key, value))

    main(args)
