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
from Preprocessing.ImageGeneratorManager import *
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
    from Networks_Pytorch.Networks import *
    from Networks_Pytorch.Optimizers import *
    from Networks_Pytorch.Trainers import *
import argparse



def main(args):
    # First thing, set session in the selected(s) devices: CPU or GPU
    set_session_in_selected_device(use_GPU_device=True,
                                   type_GPU_installed=args.typeGPUinstalled)

    # ---------- SETTINGS ----------
    nameModelsRelPath = args.modelsdir

    # Get the file list:
    nameImagesFiles = 'images*' + getFileExtension(FORMATTRAINDATA)
    nameGroundTruthFiles = 'grndtru*'+ getFileExtension(FORMATTRAINDATA)
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    TrainingDataPath = workDirsManager.getNameExistPath(workDirsManager.getNameTrainingDataPath())
    if args.use_restartModel:
        ModelsPath = workDirsManager.getNameExistPath(args.basedir, nameModelsRelPath)
    else:
        ModelsPath = workDirsManager.getNameUpdatePath(args.basedir, nameModelsRelPath)

    listTrainImagesFiles = findFilesDir(TrainingDataPath, nameImagesFiles)
    listTrainGroundTruthFiles = findFilesDir(TrainingDataPath, nameGroundTruthFiles)

    if args.useValidationData:
        ValidationDataPath = workDirsManager.getNameExistPath(workDirsManager.getNameValidationDataPath())

        listValidImagesFiles = findFilesDir(ValidationDataPath, nameImagesFiles)
        listValidGroundTruthFiles = findFilesDir(ValidationDataPath, nameGroundTruthFiles)

        if not listValidImagesFiles or not listValidGroundTruthFiles:
            use_validation_data = False
            message = "No validation data used for training the model..."
            CatchWarningException(message)
        else:
            use_validation_data = True
    else:
        use_validation_data = False


    
    # BUILDING MODEL
    # ----------------------------------------------
    print("_" * 30)
    print("Building model...")
    print("_" * 30)

    if args.use_restartModel:
        initial_epoch = args.epoch_restart
        args.num_epochs += initial_epoch
    else:
        initial_epoch = 0

    if TYPE_DNNLIBRARY_USED == 'Keras':
        if (not args.use_restartModel) or (args.use_restartModel and args.restart_only_weights):
            model_constructor = DICTAVAILMODELS3D(IMAGES_DIMS_Z_X_Y,
                                                  tailored_build_model=args.tailored_build_model,
                                                  num_layers=args.num_layers,
                                                  num_featmaps_base=args.num_featmaps_base,
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
            # output model summary
            model.summary()

            if args.use_restartModel:
                print("Loading saved weights and restarting...")
                modelSavedPath = joinpathnames(ModelsPath, 'model_'+ args.restart_modelFile +'.hdf5')
                print("Restarting from file: \'%s\'..." %(modelSavedPath))
                model.load_weights(modelSavedPath)

        else: #args.use_restartModel and args.restart_only_weights:
            print("Loading full model: weights, optimizer, loss, metrics ... and restarting...")
            modelSavedPath = joinpathnames(ModelsPath, 'model_' + args.restart_modelFile + '.hdf5')
            print("Restarting from file: \'%s\'..." %(modelSavedPath))

            loss_fun = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest).loss
            metrics  =[DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest).get_renamed_compute() for imetrics in args.listmetrics]
            custom_objects = dict(map(lambda fun: (fun.__name__, fun), [loss_fun] + metrics))
            # load and compile model
            model = NeuralNetwork.get_load_saved_model(modelSavedPath, custom_objects=custom_objects)

        # Callbacks:
        callbacks_list = []
        callbacks_list.append(RecordLossHistory(ModelsPath, [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest).get_renamed_compute() for imetrics in args.listmetrics]))
        filename = joinpathnames(ModelsPath, 'model_{epoch:02d}_{loss:.5f}_{val_loss:.5f}.hdf5')
        callbacks_list.append(callbacks.ModelCheckpoint(filename, monitor='loss', verbose=0))
        # callbacks_list.append(callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='max'))

        # output model summary
        model.summary()

    elif TYPE_DNNLIBRARY_USED == 'Pytorch':
        if (not args.use_restartModel) or (args.use_restartModel and args.restart_only_weights):
            model_net = DICTAVAILMODELS(args.imodel, IMAGES_DIMS_Z_X_Y, nfeat=args.num_featmaps_base)
            optimizer = DICTAVAILOPTIMIZERS(args.optimizer, model_net.parameters(), lr=args.learn_rate)
            loss_fun = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest)
            trainer = Trainer(model_net, optimizer, loss_fun)

            if args.use_restartModel:
                print("Loading saved weights and restarting...")
                modelSavedPath = joinpathnames(ModelsPath, 'model_'+ args.restart_modelFile +'.pt')
                print("Restarting from file: \'%s\'..." %(modelSavedPath))
                trainer.load_model_only_weights(modelSavedPath)

        else: #args.use_restartModel and args.restart_only_weights:
            print("Loading full model: weights, optimizer, loss, metrics ... and restarting...")
            modelSavedPath = joinpathnames(ModelsPath, 'model_' + args.restart_modelFile + '.pt')
            print("Restarting from file: \'%s\'..." %(modelSavedPath))
            trainer = Trainer.load_model_full(modelSavedPath)

    trainer.setup_losshistory_filepath(ModelsPath,
                                       isexists_lossfile=args.use_restartModel)
    trainer.setup_savemodel_filepath(ModelsPath,
                                     type_save_models='full_model',
                                     type_num_models_saved='only_last_epoch')
        # output model summary
        trainer.get_summary_model()
    # ----------------------------------------------



    # LOADING DATA
    # ----------------------------------------------
    print("-" * 30)
    print("Loading data...")
    print("-" * 30)

    print("Load Training data...")
    if (args.slidingWindowImages or args.transformationImages or args.elasticDeformationImages):
        print("Generate Training images with Batch Generator of Training data...")
        (train_xData, train_yData) = LoadDataManager.loadData_ListFiles(listTrainImagesFiles,
                                                                        listTrainGroundTruthFiles)
        train_images_generator = getImagesDataGenerator3D(args.slidingWindowImages,
                                                          args.prop_overlap_Z_X_Y,
                                                          args.transformationImages,
                                                          args.elasticDeformationImages)
        train_batch_data_generator = TrainingBatchDataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                train_xData,
                                                                train_yData,
                                                                train_images_generator,
                                                                batch_size=args.batch_size,
                                                                shuffle=SHUFFLETRAINDATA)
        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles),
                                                                          len(train_batch_data_generator)))
    else:
        (train_xData, train_yData) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_ListFiles(listTrainImagesFiles,
                                                                                                    listTrainGroundTruthFiles)
        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles),
                                                                          len(train_xData)))

    if use_validation_data:
        print("Load Validation data...")
        if (args.slidingWindowImages or args.transformationImages or args.elasticDeformationImages):
            print("Generate Validation images with Batch Generator of Validation data...")
            args.transformationImages = args.transformationImages and args.useTransformOnValidationData
            args.elasticDeformationImages = args.elasticDeformationImages and args.useTransformOnValidationData
            (valid_xData, valid_yData) = LoadDataManager.loadData_ListFiles(listValidImagesFiles,
                                                                            listValidGroundTruthFiles)
            valid_images_generator = getImagesDataGenerator3D(args.slidingWindowImages,
                                                              args.prop_overlap_Z_X_Y,
                                                              args.transformationImages,
                                                              args.elasticDeformationImages)
            valid_batch_data_generator = TrainingBatchDataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                    valid_xData,
                                                                    valid_yData,
                                                                    valid_images_generator,
                                                                    batch_size=args.batch_size,
                                                                    shuffle=SHUFFLETRAINDATA)
            validation_data = valid_batch_data_generator
            print("Number volumes: %s. Total Data batches generated: %s..." %(len(listValidImagesFiles),
                                                                              len(valid_batch_data_generator)))
        else:
            (valid_xData, valid_yData) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_ListFiles(listValidImagesFiles,
                                                                                                        listValidGroundTruthFiles)
            validation_data = (valid_xData, valid_yData)
            print("Number volumes: %s. Total Data batches generated: %s..." % (len(listTrainImagesFiles),
                                                                               len(valid_xData)))
    else:
        validation_data = None



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
                                callbacks=callbacks_list,
                                validation_data=validation_data,
                                shuffle=SHUFFLETRAINDATA,
                                initial_epoch=initial_epoch)
        else:
            model.fit(train_xData, train_yData,
                      batch_size=args.batch_size,
                      epochs=args.num_epochs,
                      steps_per_epoch=args.max_steps_epoch,
                      verbose=1,
                      callbacks=callbacks_list,
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
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--modelsdir', default='Models')
    parser.add_argument('--tailored_build_model', type=str2bool, default=TAILORED_BUILD_MODEL)
    parser.add_argument('--type_network', type=str, default=TYPE_NETWORK)
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--num_featmaps_base', type=int, default=NUM_FEATMAPS_BASE)
    parser.add_argument('--type_activate_hidden', type=str, default=TYPE_ACTIVATE_HIDDEN)
    parser.add_argument('--type_activate_output', type=str, default=TYPE_ACTIVATE_OUTPUT)
    parser.add_argument('--type_padding_convol', type=str, default=TYPE_PADDING_CONVOL)
    parser.add_argument('--disable_convol_pooling_lastlayer', type=str2bool, default=DISABLE_CONVOL_POOLING_LASTLAYER)
    parser.add_argument('--isUse_dropout', type=str2bool, default=ISUSE_DROPOUT)
    parser.add_argument('--isUse_batchnormalize', type=str2bool, default=ISUSE_BATCHNORMALIZE)
    parser.add_argument('--imodel', default=IMODEL)
    parser.add_argument('--optimizer', default=IOPTIMIZER)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--max_steps_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learn_rate', type=float, default=LEARN_RATE)
    parser.add_argument('--lossfun', default=ILOSSFUN)
    parser.add_argument('--listmetrics', type=parseListarg, default=LISTMETRICS)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=ELASTICDEFORMATIONIMAGES)
    parser.add_argument('--useValidationData', type=str2bool, default=USEVALIDATIONDATA)
    parser.add_argument('--useTransformOnValidationData', type=str2bool, default=USETRANSFORMONVALIDATIONDATA)
    parser.add_argument('--typeGPUinstalled', type=str, default=TYPEGPUINSTALLED)
    parser.add_argument('--useMultiThreading', type=str2bool, default=USEMULTITHREADING)
    parser.add_argument('--use_restartModel', type=str2bool, default=USE_RESTARTMODEL)
    parser.add_argument('--restart_modelFile', default=RESTART_MODELFILE)
    parser.add_argument('--restart_only_weights', type=str2bool, default=RESTART_ONLY_WEIGHTS)
    parser.add_argument('--epoch_restart', type=int, default=EPOCH_RESTART)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
