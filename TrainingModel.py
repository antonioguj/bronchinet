#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from CommonUtil.CPUGPUdevicesManager import *
from CommonUtil.ErrorMessages import *
from CommonUtil.ImageGeneratorManager import *
from CommonUtil.LoadDataManager import *
from CommonUtil.WorkDirsManager import *
if TYPE_DNNLIBRARY_USED == 'Keras':
    from CommonUtil.BatchDataGenerator_Keras import *
    from Networks_Keras.Callbacks import *
    from Networks_Keras.Metrics import *
    from Networks_Keras.Networks import *
    from Networks_Keras.Optimizers import *
elif TYPE_DNNLIBRARY_USED == 'Pytorch':
    from CommonUtil.BatchDataGenerator_Pytorch import *
    from CommonUtil.TrainManager_Pytorch import *
    from Networks_Pytorch.Metrics import *
    from Networks_Pytorch.Networks import *
    from Networks_Pytorch.Optimizers import *
import argparse



def main(args):

    # First thing, set session in the selected(s) devices: CPU or GPU
    set_session_in_selected_device(use_GPU_device=True,
                                   type_GPU_installed=args.typeGPUinstalled)

    # ---------- SETTINGS ----------
    if args.use_restartModel:
        nameModelsRelPath = 'Models_Restart'
    else:
        nameModelsRelPath = 'Models'

    # Get the file list:
    nameImagesFiles = 'images*'+ getFileExtension(FORMATINOUTDATA)
    nameMasksFiles  = 'masks*' + getFileExtension(FORMATINOUTDATA)
    # ---------- SETTINGS ----------


    workDirsManager    = WorkDirsManager(args.basedir)
    TrainingDataPath   = workDirsManager.getNameExistPath(workDirsManager.getNameTrainingDataPath())
    ValidationDataPath = workDirsManager.getNameExistPath(workDirsManager.getNameValidationDataPath())
    if args.use_restartModel:
        ModelsPath     = workDirsManager.getNameExistPath(args.basedir, nameModelsRelPath)
    else:
        ModelsPath     = workDirsManager.getNameUpdatePath(args.basedir, nameModelsRelPath)

    listTrainImagesFiles = findFilesDir(TrainingDataPath,   nameImagesFiles)
    listTrainMasksFiles  = findFilesDir(TrainingDataPath,   nameMasksFiles )
    listValidImagesFiles = findFilesDir(ValidationDataPath, nameImagesFiles)
    listValidMasksFiles  = findFilesDir(ValidationDataPath, nameMasksFiles )

    if not listValidImagesFiles or not listValidMasksFiles:
        use_validation_data = False
        message = "No Validation Data used for training network..."
        CatchWarningException(message)
    else:
        use_validation_data = True


    
    # BUILDING MODEL
    # ----------------------------------------------
    print("_" * 30)
    print("Building model...")
    print("_" * 30)

    if (args.multiClassCase):
        num_classes_out = args.numClassesMasks + 1
    else:
        num_classes_out = 1

    if args.use_restartModel:
        initial_epoch   = args.epoch_restart
        args.num_epochs += initial_epoch
    else:
        initial_epoch = 0

    if args.use_restartModel and not args.restart_only_weights:
        print("Loading full saved model: weights, optimizer, loss, metrics, ... and restarting...")

        modelSavedPath = joinpathnames(ModelsPath, getSavedModelFileName(args.restart_modelFile))

        if TYPE_DNNLIBRARY_USED == 'Keras':
            print("Restarting from file: \'%s\'..." %(modelSavedPath))

            train_model_funs = [DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest)] \
                               + [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest, set_fun_name=True) for imetrics in args.listmetrics]
            custom_objects = dict(map(lambda fun: (fun.__name__, fun), train_model_funs))

            model = NeuralNetwork.get_load_saved_model(modelSavedPath,
                                                   custom_objects=custom_objects)
        elif TYPE_DNNLIBRARY_USED == 'Pytorch':
            print("Restarting from file not implemented yet...")

    else:
        if TYPE_DNNLIBRARY_USED == 'Keras':
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
                                                  isuse_batchnormalize=args.isUse_batchnormalize,
                                                  num_classes_out=num_classes_out)
            model = model_constructor.get_model()

            # Compile model
            model.compile(optimizer= DICTAVAILOPTIMIZERS(args.optimizer, lr=args.learn_rate),
                          loss     = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest),
                          metrics  =[DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest, set_fun_name=True) for imetrics in args.listmetrics])

            if args.use_restartModel and args.restart_only_weights:
                print("Loading saved weights and restarting...")

                modelSavedPath = joinpathnames(ModelsPath, getSavedModelFileName(args.restart_modelFile))

                print("Restarting from file: \'%s\'..." % (modelSavedPath))

                model.load_weights(modelSavedPath)

            model.summary()

            # Callbacks:
            callbacks_list = []
            callbacks_list.append(RecordLossHistory(ModelsPath, [
                DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest, set_fun_name=True) for
                imetrics in args.listmetrics]))

            filename = ModelsPath + '/model_{epoch:02d}_{loss:.5f}_{val_loss:.5f}.hdf5'
            callbacks_list.append(callbacks.ModelCheckpoint(filename, monitor='loss', verbose=0))
            # callbacks_list.append(callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='max'))

        elif TYPE_DNNLIBRARY_USED == 'Pytorch':
            model_net = DICTAVAILMODELS3D(IMAGES_DIMS_Z_X_Y)

            optimizer = DICTAVAILOPTIMIZERS(args.optimizer, model_net.parameters(), lr=args.learn_rate),
            loss_fun  = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest),
            metrics   =[DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest, set_fun_name=True) for imetrics in args.listmetrics]

            if args.use_restartModel and args.restart_only_weights:
                print("Restarting from file not implemented yet...")

            train_manager = TrainManager(model_net, optimizer, loss_fun, metrics)
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
                                                                        listTrainMasksFiles)

        train_images_generator = getImagesDataGenerator3D(args.slidingWindowImages,
                                                          args.prop_overlap_Z_X_Y,
                                                          args.transformationImages,
                                                          args.elasticDeformationImages)

        train_batch_data_generator = TrainingBatchDataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                train_xData,
                                                                train_yData,
                                                                train_images_generator,
                                                                num_classes_out=num_classes_out,
                                                                batch_size=args.batch_size,
                                                                shuffle=True)

        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles),
                                                                          len(train_batch_data_generator)))
    else:
        (train_xData, train_yData) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_ListFiles(listTrainImagesFiles,
                                                                                                    listTrainMasksFiles)

        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles),
                                                                          len(train_xData)))


    if use_validation_data:
        print("Load Validation data...")
        if (args.slidingWindowImages or args.transformationImages or args.elasticDeformationImages):
            print("Generate Validation images with Batch Generator of Validation data...")

            args.transformationImages     = args.transformationImages and args.useTransformOnValidationData
            args.elasticDeformationImages = args.elasticDeformationImages and args.useTransformOnValidationData

            (valid_xData, valid_yData) = LoadDataManager.loadData_ListFiles(listValidImagesFiles,
                                                                            listValidMasksFiles)

            valid_images_generator = getImagesDataGenerator3D(args.slidingWindowImages,
                                                              args.prop_overlap_Z_X_Y,
                                                              args.transformationImages,
                                                              args.elasticDeformationImages)

            valid_batch_data_generator = TrainingBatchDataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                    valid_xData,
                                                                    valid_yData,
                                                                    valid_images_generator,
                                                                    num_classes_out=num_classes_out,
                                                                    batch_size=args.batch_size,
                                                                    shuffle=True)
            validation_data = valid_batch_data_generator

            print("Number volumes: %s. Total Data batches generated: %s..." %(len(listValidImagesFiles),
                                                                              len(valid_batch_data_generator)))
        else:
            (valid_xData, valid_yData) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_ListFiles(listValidImagesFiles,
                                                                                                        listValidMasksFiles)
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
                                steps_per_epoch=len(train_batch_data_generator),
                                nb_epoch=args.num_epochs,
                                verbose=1,
                                callbacks=callbacks_list,
                                validation_data=validation_data,
                                shuffle=True,
                                initial_epoch=initial_epoch)
        else:
            model.fit(train_xData, train_yData,
                      batch_size=args.batch_size,
                      epochs=args.num_epochs,
                      verbose=1,
                      callbacks=callbacks_list,
                      validation_data=validation_data,
                      shuffle=True,
                      initial_epoch=initial_epoch)

    elif TYPE_DNNLIBRARY_USED == 'Pytorch':
        train_manager.train(train_batch_data_generator,
                            valid_batch_data_generator,
                            num_epochs=args.num_epochs,
                            initial_epoch=initial_epoch)
    # ----------------------------------------------



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
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
    parser.add_argument('--optimizer', default=IOPTIMIZER)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learn_rate', type=float, default=LEARN_RATE)
    parser.add_argument('--lossfun', default=ILOSSFUN)
    parser.add_argument('--listmetrics', type=parseListarg, default=LISTMETRICS)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=ELASTICDEFORMATIONIMAGES)
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