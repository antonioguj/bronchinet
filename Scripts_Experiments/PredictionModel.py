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
from Preprocessing.ImageGeneratorManager import *
from Postprocessing.ImageReconstructorManager import *
from DataLoaders.LoadDataManager import *
if TYPE_DNNLIBRARY_USED == 'Keras':
    from DataLoaders.BatchDataGenerator_Keras import *
    from Networks_Keras.Metrics import *
    from Networks_Keras.Networks import *
    from Networks_Keras.VisualModelParams import *
elif TYPE_DNNLIBRARY_USED == 'Pytorch':
    from DataLoaders.BatchDataGenerator_Pytorch import *
    from Networks_Pytorch.Trainers import *
    from Networks_Pytorch.Metrics import *
    from Networks_Pytorch.Networks import *
from Preprocessing.OperationImages import *
from Preprocessing.OperationMasks import *
import argparse



def main(args):
    # First thing, set session in the selected(s) devices: CPU or GPU
    set_session_in_selected_device(use_GPU_device=True,
                                   type_GPU_installed=args.typeGPUinstalled)

    # ---------- SETTINGS ----------
    nameInputRoiMasksRelPath   = 'Lungs_Proc/'
    nameInputReferFilesRelPath = 'Images_Proc/'
    nameImagesFiles            = 'images*' + getFileExtension(FORMATTRAINDATA)
    nameLabelsFiles            = 'labels*' + getFileExtension(FORMATTRAINDATA)
    nameInputRoiMasksFiles     = '*_lungs.nii.gz'
    nameInputReferFiles        = '*.nii.gz'
    nameRescaleFactors         = 'rescaleFactors_images.npy'
    nameCropBoundingBoxes      = 'cropBoundingBoxes_images.npy'
    nameOutputPredictionFiles  = 'predict-probmaps_%s.nii.gz'

    if (args.saveFeatMapsLayers):
        nameOutputFeatureMapsDirs = 'featureMaps-%s_lay_%s'
        nameOutputFeatureMapsFiles = 'featmaps-%s_lay_%s_map%0.2i.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager     = WorkDirsManager(args.basedir)
    TestingDataPath     = workDirsManager.getNameExistPath(args.testdatadir)
    InputReferFilesPath = workDirsManager.getNameExistBaseDataPath(nameInputReferFilesRelPath)
    ModelsPath          = workDirsManager.getNameExistPath(args.modelsdir)
    OutputPredictionPath= workDirsManager.getNameNewPath  (args.predictionsdir)

    listTestImagesFiles = findFilesDirAndCheck(TestingDataPath,     nameImagesFiles)
    listTestLabelsFiles = findFilesDirAndCheck(TestingDataPath,     nameLabelsFiles)
    listInputReferFiles = findFilesDirAndCheck(InputReferFilesPath, nameInputReferFiles)

    if (args.masksToRegionInterest):
        InputRoiMasksPath      = workDirsManager.getNameExistBaseDataPath(nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath, nameInputRoiMasksFiles)

    if (args.rescaleImages):
        dict_rescaleFactors = readDictionary(joinpathnames(workDirsManager.getNameBaseDataPath(), nameRescaleFactors))

    if (args.cropImages):
        dict_cropBoundingBoxes = readDictionary(joinpathnames(workDirsManager.getNameBaseDataPath(), nameCropBoundingBoxes))


    test_images_generator = getImagesDataGenerator3D(args.slidingWindowImages,
                                                     args.prop_overlap_Z_X_Y,
                                                     args.transformationImages,
                                                     args.elasticDeformationImages)

    images_reconstructor = getImagesReconstructor3D(args.slidingWindowImages,
                                                    args.prop_overlap_Z_X_Y,
                                                    use_TransformationImages=False,
                                                    isfilterImages=args.filterPredictProbMaps,
                                                    prop_valid_outUnet=args.prop_valid_outUnet)



    # LOADING MODEL
    # ----------------------------------------------
    print("-" * 30)
    print("Loading saved model...")
    print("-" * 30)

    if TYPE_DNNLIBRARY_USED == 'Keras':
        print("Loading full model: weights, optimizer, loss, metrics ... and restarting...")
        modelSavedPath = joinpathnames(ModelsPath, 'model_' + args.prediction_modelFile + '.hdf5')
        print("Restarting from file: \'%s\'..." % (modelSavedPath))

        loss_fun = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest).loss
        metrics = [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest).get_renamed_compute() for imetrics in args.listmetrics]
        custom_objects = dict(map(lambda fun: (fun.__name__, fun), [loss_fun] + metrics))
        # load and compile model
        model = NeuralNetwork.get_load_saved_model(modelSavedPath, custom_objects=custom_objects)

        # output model summary
        model.summary()

    elif TYPE_DNNLIBRARY_USED == 'Pytorch':
        print("Loading full model: weights, optimizer, loss, metrics ... and restarting...")
        modelSavedPath = joinpathnames(ModelsPath, 'model_' + args.prediction_modelFile + '.pt')
        print("Restarting from file: \'%s\'..." % (modelSavedPath))
        # load and compile model
        trainer = Trainer.load_model_full(modelSavedPath)

        # output model summary
        trainer.get_summary_model()


    if (args.saveFeatMapsLayers):
        if TYPE_DNNLIBRARY_USED == 'Keras':
            visual_model_params = VisualModelParams(model, IMAGES_DIMS_Z_X_Y)
            if args.firstSaveFeatMapsLayers:
                get_index_featmap = lambda i: args.firstSaveFeatMapsLayers + i
            else:
                get_index_featmap = lambda i: i

        elif TYPE_DNNLIBRARY_USED == 'Pytorch':
            message = 'Visualize a model feature maps still not implemented...'
            CatchErrorException(message)
    # ----------------------------------------------



    # START ANALYSIS
    # ----------------------------------------------
    print("-" * 30)
    print("Predicting model...")
    print("-" * 30)

    for ifile, test_xData_file in enumerate(listTestImagesFiles):
        print("\nInput: \'%s\'..." % (basename(test_xData_file)))


        # COMPUTE PREDICTION
        # ------------------------------------------
        print("Loading data...")
        if (args.slidingWindowImages or args.transformationImages):
            if TYPE_DNNLIBRARY_USED == 'Keras':
                test_xData = LoadDataManagerInBatches_DataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                    test_images_generator).loadData_1File(test_xData_file,
                                                                                                          shuffle_images=False)
            elif TYPE_DNNLIBRARY_USED == 'Pytorch':
                test_xData = LoadDataManager.loadData_1File(test_xData_file)
                test_batch_data_generator = TrainingBatchDataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                       [test_xData],
                                                                       [test_xData],
                                                                       test_images_generator,
                                                                       batch_size=1,
                                                                       shuffle=False)
                (test_yData, test_xData) = DataSampleGenerator(IMAGES_DIMS_Z_X_Y,
                                                               [test_xData],
                                                               [test_xData],
                                                               test_images_generator).get_full_data()
        else:
            test_xData = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_1File(test_xData_file)
            test_xData = np.expand_dims(test_xData, axis=0)

        print("Total Data batches generated: %s..." % (len(test_xData)))


        print("Evaluate model...")
        if TYPE_DNNLIBRARY_USED == 'Keras':
            predict_yData = model.predict(test_xData,
                                          batch_size=1)
        elif TYPE_DNNLIBRARY_USED == 'Pytorch':
            predict_yData = trainer.predict(test_batch_data_generator)


        if (args.saveFeatMapsLayers):
            print("Compute feature maps of evaluated model...")
            featuremaps_data = visual_model_params.get_feature_maps(test_xData, args.nameSaveModelLayer,
                                                                         max_num_feat_maps=args.maxNumSaveFeatMapsLayers,
                                                                         first_feat_maps=args.firstSaveFeatMapsLayers)
        # ------------------------------------------



        # RECONSTRUCT FULL-SIZE PREDICTION
        # ------------------------------------------
        print("Reconstruct prediction to full size...")
        # Assign original images and masks files
        index_referimg_file = getIndexOriginImagesFile(basename(test_xData_file), beginString='images', firstIndex='01')
        in_referimage_file = listInputReferFiles[index_referimg_file]
        print("Reference image file: \'%s\'..." %(basename(in_referimage_file)))

        # init reconstructor with size of "ifile"
        predict_fullsize_shape = FileReader.getImageSize(test_xData_file)
        images_reconstructor.complete_init_data(predict_fullsize_shape)

        prediction_array = images_reconstructor.compute(predict_yData)

        if (args.saveFeatMapsLayers):
            featuremaps_array = images_reconstructor.compute(featuremaps_data)


        # reconstruct from cropped / rescaled images
        referimage_shape = FileReader.getImageSize(in_referimage_file)

        if (args.cropImages):
            crop_bounding_box = dict_cropBoundingBoxes[filenamenoextension(in_referimage_file)]
            print("Predicted data are cropped. Extend array size to original. Bounding-box: \'%s\'..." %(str(crop_bounding_box)))

            prediction_array = ExtendImages.compute3D(prediction_array, crop_bounding_box, referimage_shape)
            print("Final dims: %s..." %(str(prediction_array.shape)))


        if (args.masksToRegionInterest):
            print("Mask predictions to RoI: lungs...")
            in_roimask_file = listInputRoiMasksFiles[index_referimg_file]
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

            roimask_array = FileReader.getImageArray(in_roimask_file)
            prediction_array = OperationBinaryMasks.reverse_mask_exclude_voxels_fillzero(prediction_array, roimask_array)


        if (args.saveFeatMapsLayers):
            print("Reconstruct predicted feature maps to full size...")
            if (args.cropImages):
                num_featmaps = featuremaps_array.shape[-1]
                featuremaps_shape = list(referimage_shape) + [num_featmaps]
                featuremaps_array = ExtendImages.compute3D(featuremaps_array, crop_bounding_box, featuremaps_shape)

            if (args.masksToRegionInterest):
                featuremaps_array = OperationBinaryMasks.reverse_mask_exclude_voxels_fillzero(featuremaps_array, roimask_array)
        # ------------------------------------------



        out_prediction_file = joinpathnames(OutputPredictionPath, nameOutputPredictionFiles %(filenamenoextension(in_referimage_file)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_prediction_file), prediction_array.shape))

        FileReader.writeImageArray(out_prediction_file, prediction_array)

        if (args.saveFeatMapsLayers):
            nameOutputFeatureMapsRelPath = nameOutputFeatureMapsDirs %(filenamenoextension(in_referimage_file), args.nameSaveModelLayer)
            OutputFeatureMapsPath = workDirsManager.getNameNewPath(OutputPredictionPath, nameOutputFeatureMapsRelPath)

            num_featmaps = featuremaps_array.shape[-1]
            for ifeatmap in range(num_featmaps):
                out_featuremaps_file = joinpathnames(OutputFeatureMapsPath, nameOutputFeatureMapsFiles %(filenamenoextension(in_referimage_file),
                                                                                                         args.nameSaveModelLayer,
                                                                                                         get_index_featmap(ifeatmap)+1))
                print("Output: \'%s\', of dims \'%s\'..." %(basename(out_featuremaps_file), featuremaps_array[...,ifeatmap].shape))

                FileReader.writeImageArray(out_featuremaps_file, featuremaps_array[...,ifeatmap])
            #endfor
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--testdatadir', default='TestingData')
    parser.add_argument('--typedata', default=TYPEDATAPREDICT)
    parser.add_argument('--modelsdir', default='Models_Restart')
    parser.add_argument('--predictionsdir', default='Predictions_NEW')
    parser.add_argument('--lossfun', default=ILOSSFUN)
    parser.add_argument('--listmetrics', type=parseListarg, default=LISTMETRICS)
    parser.add_argument('--prediction_modelFile', default=PREDICTION_MODELFILE)
    parser.add_argument('--filterPredictProbMaps', type=str2bool, default=FILTERPREDICTPROBMAPS)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--rescaleImages', type=str2bool, default=False)
    parser.add_argument('--prop_valid_outUnet', type=float, default=PROP_VALID_OUTUNET)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--extendSizeImages', type=str2bool, default=EXTENDSIZEIMAGES)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=False)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=False)
    parser.add_argument('--typeGPUinstalled', type=str, default=TYPEGPUINSTALLED)
    parser.add_argument('--saveFeatMapsLayers', type=str2bool, default=SAVEFEATMAPSLAYERS)
    parser.add_argument('--nameSaveModelLayer', default=NAMESAVEMODELLAYER)
    parser.add_argument('--maxNumSaveFeatMapsLayers', type=int, default=None)
    parser.add_argument('--firstSaveFeatMapsLayers', type=int, default=None)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
