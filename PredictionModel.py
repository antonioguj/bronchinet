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
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.ImageGeneratorManager import *
from CommonUtil.ImageReconstructorManager import *
from CommonUtil.LoadDataManager import *
from CommonUtil.PlotsManager import *
from CommonUtil.WorkDirsManager import *
from Networks_Keras.Metrics import *
from Networks_Keras.Networks import *
from Networks_Keras.VisualModelParams import *
from Preprocessing.BoundingBoxMasks import *
from Preprocessing.OperationsImages import *
from Preprocessing.OperationsMasks import *
import argparse



def main(args):

    # First thing, set session in the selected(s) devices: CPU or GPU
    set_session_in_selected_device(use_GPU_device=True,
                                   type_GPU_installed=args.typeGPUinstalled)

    # ---------- SETTINGS ----------
    nameInputImagesRelPath = 'ProcImages'
    nameInputMasksRelPath  = 'ProcMasks'
    nameLungsMasksRelPath  = 'ProcAllMasks'

    # Get the file list:
    nameImagesFiles = 'images*'+ getFileExtension(FORMATINOUTDATA)
    nameMasksFiles  = 'masks*' + getFileExtension(FORMATINOUTDATA)

    nameInputImagesFiles = '*.nii.gz'
    nameInputMasksFiles  = '*.nii.gz'
    nameLungsMasksFiles  = '*_lungs.nii.gz'

    nameBoundingBoxesMasks = 'boundBoxesMasks.npy'

    tempNamePredictMasksFiles = 'predict_probmaps-%s.nii.gz'

    if (args.saveFeatMapsLayers):
        tempNameSaveFeatMapsDirs  = 'featureMaps-%s-lay_%s'
        tempNameSaveFeatMapsFiles = 'featmaps-%s-lay_%s-map%0.2i.nii.gz'

    if (args.savePredictMaskSlices):
        tempNameSavePredictMaskSliceImagesDirs = 'imagesSlices-%s'
        if (args.saveFeatMapsLayers):
            tempNameSaveFeatMapsSliceImagesDirs = 'imagesSlices-%s-lay_%s-map%0.2i'
    # ---------- SETTINGS ----------



    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath    = workDirsManager.getNameBaseDataPath()
    TestingDataPath = workDirsManager.getNameExistPath(workDirsManager.getNameDataPath(args.typedata))
    ModelsPath      = workDirsManager.getNameExistPath(args.basedir, args.modelsdir)
    PredictDataPath = workDirsManager.getNameNewPath  (args.basedir, args.predictionsdir)

    listTestImagesDataFiles = findFilesDir(TestingDataPath, nameImagesFiles)
    listTestMasksDataFiles  = findFilesDir(TestingDataPath, nameMasksFiles )

    if (args.constructInputDataDLCST):
        listTestImagesDataFiles_2 = findElementsSubstringInListStrings(listTestImagesDataFiles, 'img2')
        listTestMasksDataFiles_2  = findElementsSubstringInListStrings(listTestMasksDataFiles,  'img2')

        listTestImagesDataFiles = findElementsSubstringInListStrings(listTestImagesDataFiles, 'img1')
        listTestMasksDataFiles  = findElementsSubstringInListStrings(listTestMasksDataFiles,  'img1')

    InputImagesPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputImagesRelPath)
    InputMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, nameInputMasksRelPath )

    listFullImagesFiles     = findFilesDir(InputImagesPath, nameInputImagesFiles)
    listGrndTruthMasksFiles = findFilesDir(InputMasksPath,  nameInputMasksFiles)


    test_images_generator = getImagesDataGenerator3D(args.slidingWindowImages,
                                                     args.prop_overlap_Z_X_Y,
                                                     args.transformationImages,
                                                     args.elasticDeformationImages)

    images_reconstructor = getImagesReconstructor3D(args.slidingWindowImages,
                                                    args.prop_overlap_Z_X_Y,
                                                    use_TransformationImages=False,
                                                    isfilterImages=args.filterPredictProbMaps,
                                                    prop_valid_outUnet=args.prop_valid_outUnet)

    if (args.masksToRegionInterest):
        LungsMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameLungsMasksRelPath)

        listLungsMasksFiles = findFilesDir(LungsMasksPath, nameLungsMasksFiles)

    if (args.cropImages or args.extendSizeImages):
        namefile_dict = joinpathnames(BaseDataPath, nameBoundingBoxesMasks)
        dict_masks_boundingBoxes = readDictionary(namefile_dict)

    if (args.multiClassCase):
        num_classes_out = args.numClassesMasks + 1
    else:
        num_classes_out = 1



    print("-" * 30)
    print("Loading saved model...")
    print("-" * 30)

    # Loading Saved Model
    modelSavedPath = joinpathnames(ModelsPath, getSavedModelFileName(args.prediction_modelFile))

    train_model_funs = [DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest)] \
                       + [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest, set_fun_name=True) for imetrics in args.listmetrics]
    custom_objects = dict(map(lambda fun: (fun.__name__, fun), train_model_funs))

    model = NeuralNetwork.get_load_saved_model(modelSavedPath, custom_objects=custom_objects)

    if (args.saveFeatMapsLayers):
        visual_model_params = VisualModelParams(model, IMAGES_DIMS_Z_X_Y)

        if args.firstSaveFeatMapsLayers:
            get_index_featmap = lambda i: args.firstSaveFeatMapsLayers+i
        else:
            get_index_featmap = lambda i: i

    computePredictAccuracy = DICTAVAILMETRICFUNS(args.predictAccuracyMetrics, use_in_Keras=False)



    # START ANALYSIS
    # ------------------------------
    print("-" * 30)
    print("Predicting model...")
    print("-" * 30)

    for ifile, (test_xData_file, test_yData_file) in enumerate(zip(listTestImagesDataFiles, listTestMasksDataFiles)):

        print('\'%s\'...' % (test_xData_file))

        # Assign original images and masks files
        index_input_images = getIndexOriginImagesFile(basename(test_xData_file), beginString='images', firstIndex='01')

        full_images_file    = listFullImagesFiles    [index_input_images]
        grndtruth_masks_file= listGrndTruthMasksFiles[index_input_images]

        print("assigned to original files: '%s' and '%s'..." %(basename(full_images_file), basename(grndtruth_masks_file)))


        # LOADING DATA
        print("Loading data...")
        if (args.slidingWindowImages or args.transformationImages):
            (test_xData, test_yData) = LoadDataManagerInBatches_DataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                              test_images_generator,
                                                                              num_classes_out=num_classes_out).loadData_1File(test_xData_file,
                                                                                                                              test_yData_file,
                                                                                                                              shuffle_images=False)
        else:
            (test_xData, test_yData) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_1File(test_xData_file,
                                                                                                  test_yData_file,
                                                                                                  shuffle_images=False)
            test_xData = np.expand_dims(test_xData, axis=0)
            test_yData = np.expand_dims(test_yData, axis=0)



        # EVALUATE MODEL
        print("Evaluate model...")
        predict_yData = model.predict(test_xData, batch_size=1)

        accuracy = computePredictAccuracy(test_yData.astype(FORMATPROBABILITYDATA), predict_yData)
        print("Computed accuracy: %s..." % (accuracy))

        if (args.saveFeatMapsLayers):
            print("Compute feature maps of evaluated model...")
            predict_featmaps_data = visual_model_params.get_feature_maps(test_xData, args.nameSaveModelLayer,
                                                                         max_num_feat_maps=args.maxNumSaveFeatMapsLayers,
                                                                         first_feat_maps=args.firstSaveFeatMapsLayers)

        # RECONSTRUCT FULL PREDICTED PROBABILITY MAPS
        print("Reconstruct full predicted data from sliding-window batches...")
        fullsize_ydata_shape = FileReader.getImageSize(test_yData_file)

        # init reconstructor with size of "ifile"
        images_reconstructor.complete_init_data(fullsize_ydata_shape)

        predict_probmaps_array = images_reconstructor.compute(predict_yData)

        if (args.saveFeatMapsLayers):
            predict_featmaps_array = images_reconstructor.compute(predict_featmaps_data)



        # **********************************************************************************
        # FOR DLCST DATA: SPLIT EACH CT IN TWO LUNGS. ANALYSE SECOND LUNG. RECONSTRUCT LATER
        if (args.constructInputDataDLCST):
            print("Construct data for DLCST: crop images and set one batch per image...")

            test_xData_file_2 = listTestImagesDataFiles_2[ifile]
            test_yData_file_2 = listTestMasksDataFiles_2 [ifile]

            print("assigned to original files: '%s' and '%s'..." % (basename(test_xData_file_2), basename(test_yData_file_2)))


            # LOADING DATA
            print("Loading data (2nd batch)...")
            if (args.slidingWindowImages or args.transformationImages):
                (test_xData, test_yData) = LoadDataManagerInBatches_DataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                                  test_images_generator,
                                                                                  num_classes_out=num_classes_out).loadData_1File(test_xData_file_2,
                                                                                                                                  test_yData_file_2,
                                                                                                                                  shuffle_images=False)
            else:
                (test_xData, test_yData) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_1File(test_xData_file_2,
                                                                                                      test_yData_file_2,
                                                                                                      shuffle_images=False)
                test_xData = np.expand_dims(test_xData, axis=0)
                test_yData = np.expand_dims(test_yData, axis=0)



            # EVALUATE MODEL
            print("Evaluate model (2nd batch)...")
            predict_yData_2 = model.predict(test_xData, batch_size=1)

            accuracy = computePredictAccuracy(test_yData.astype(FORMATPROBABILITYDATA), predict_yData_2)
            print("Computed accuracy: %s..." %(accuracy))

            if (args.saveFeatMapsLayers):
                print("Compute feature maps of evaluated model...")
                predict_featmaps_data_2 = visual_model_params.get_feature_maps(test_xData, args.nameSaveModelLayer,
                                                                               max_num_feat_maps=args.maxNumSaveFeatMapsLayers,
                                                                               first_feat_maps=args.firstSaveFeatMapsLayers)


            # RECONSTRUCT FULL PREDICTED PROBABILITY MAPS
            print("Reconstruct full predicted data from sliding-window batches...")
            fullsize_ydata_shape = FileReader.getImageSize(test_yData_file_2)

            # init reconstructor with size of "ifile"
            images_reconstructor.complete_init_data(fullsize_ydata_shape)

            predict_probmaps_array_2 = images_reconstructor.compute(predict_yData_2)

            if (args.saveFeatMapsLayers):
                predict_featmaps_array_2 = images_reconstructor.compute(predict_featmaps_data_2)
        # **********************************************************************************



        # RECONSTRUCT FULL SIZE IMAGE FROM CROPPED IMAGES
        if (args.constructInputDataDLCST):
            print("Reconstruct data for DLCST: two cropped batches one per lung...")

            bounding_box = dict_masks_boundingBoxes[filenamenoextension(full_images_file)]

            (bounding_box_left, bounding_box_right) = BoundingBoxMasks.compute_split_bounding_boxes(bounding_box, axis=2)

            grndtruth_masks_array_shape = FileReader.getImageSize(grndtruth_masks_file)

            predict_probmaps_array = ExtendImages.compute3D(predict_probmaps_array,
                                                            bounding_box_left,
                                                            grndtruth_masks_array_shape)
            IncludeImages.compute3D(predict_probmaps_array_2,
                                    bounding_box_right,
                                    predict_probmaps_array)

            if (args.saveFeatMapsLayers):
                num_save_featmaps = predict_featmaps_array.shape[-1]
                predict_featmaps_array_fullshape = list(grndtruth_masks_array_shape) + [num_save_featmaps]

                predict_featmaps_array = ExtendImages.compute3D(predict_featmaps_array,
                                                                bounding_box_left,
                                                                predict_featmaps_array_fullshape)
                IncludeImages.compute3D(predict_featmaps_array_2,
                                        bounding_box_right,
                                        predict_featmaps_array)
        else:
            grndtruth_masks_array_shape = FileReader.getImageSize(grndtruth_masks_file)

            if (args.cropImages):
                print("Predicted data are cropped. Extend array size from %s to original %s..."%(predict_probmaps_array.shape,
                                                                                                         grndtruth_masks_array_shape))
                bounding_box = dict_masks_boundingBoxes[filenamenoextension(full_images_file)]

                predict_probmaps_array = ExtendImages.compute3D(predict_probmaps_array,
                                                                bounding_box,
                                                                grndtruth_masks_array_shape)

                if (args.saveFeatMapsLayers):
                    num_save_featmaps = predict_featmaps_array.shape[-1]
                    predict_featmaps_array_fullshape = list(grndtruth_masks_array_shape) + [num_save_featmaps]

                    predict_featmaps_array = ExtendImages.compute3D(predict_featmaps_array,
                                                                    bounding_box,
                                                                    predict_featmaps_array_fullshape)

            elif (args.extendSizeImages):
                print("Predicted data are extended. Crop array size from %s to original %s..."%(predict_probmaps_array.shape,
                                                                                                grndtruth_masks_array_shape))
                bounding_box = dict_masks_boundingBoxes[filenamenoextension(full_images_file)]

                predict_probmaps_array = CropImages.compute3D(predict_probmaps_array,
                                                              bounding_box)

                if (args.saveFeatMapsLayers):
                    predict_featmaps_array = CropImages.compute3D(predict_featmaps_array,
                                                                  bounding_box)

            else:
                if (predict_probmaps_array.shape != grndtruth_masks_array_shape):
                    message = "size of predicted data: %s, not equal to size of ground-truth: %s..." %(predict_probmaps_array.shape,
                                                                                                       grndtruth_masks_array_shape)
                    CatchErrorException(message)



        # MASKS PREDICTED DATA TO REGION OF INTEREST
        if (args.masksToRegionInterest):
            print("Mask predicted data to Region of Interest: lungs...")
            lungs_masks_file = listLungsMasksFiles[index_input_images]

            print("assigned to: '%s'..." %(basename(lungs_masks_file)))

            lungs_masks_array = FileReader.getImageArray(lungs_masks_file)

            predict_probmaps_array = OperationsBinaryMasks.reverse_mask_exclude_voxels_fillzero(predict_probmaps_array,
                                                                                                lungs_masks_array)
            if (args.saveFeatMapsLayers):
                OperationsBinaryMasks.reverse_mask_exclude_voxels_fillzero(predict_featmaps_array, lungs_masks_array)



        # SAVE RECONSTRUCTED PREDICTED DATA
        print("Saving predicted probability maps, with dims: %s..." %(tuple2str(predict_probmaps_array.shape)))

        out_predictMasksFilename = tempNamePredictMasksFiles %(filenamenoextension(full_images_file))
        out_predictMasksFilename = joinpathnames(PredictDataPath, out_predictMasksFilename)

        FileReader.writeImageArray(out_predictMasksFilename, predict_probmaps_array)

        if (args.saveFeatMapsLayers):
            print("Saving computed %s feature maps, with dims: %s..." %(predict_featmaps_array.shape[-1],
                                                                        tuple2str(predict_featmaps_array.shape[0:-1])))

            SaveFeatMapsRelPath = tempNameSaveFeatMapsDirs %(filenamenoextension(full_images_file), args.nameSaveModelLayer)
            SaveFeatMapsPath    = workDirsManager.getNameNewPath(PredictDataPath, SaveFeatMapsRelPath)

            num_save_featmaps = predict_featmaps_array.shape[-1]

            for ifeatmap in range(num_save_featmaps):
                ind_featmap = get_index_featmap(ifeatmap)
                out_featMapsFilename = tempNameSaveFeatMapsFiles %(filenamenoextension(full_images_file), args.nameSaveModelLayer, ind_featmap+1)
                out_featMapsFilename = joinpathnames(SaveFeatMapsPath, out_featMapsFilename)

                FileReader.writeImageArray(out_featMapsFilename, predict_featmaps_array[...,ifeatmap])
            #endfor


        # SAVE PREDICTIOSN IN IMAGES
        if (args.savePredictMaskSlices):
            SaveImagesRelPath = tempNameSavePredictMaskSliceImagesDirs %(filenamenoextension(full_images_file))
            SaveImagesPath    = workDirsManager.getNameNewPath(PredictDataPath, SaveImagesRelPath)

            full_images_array    = FileReader.getImageArray(full_images_file)
            grndtruth_masks_array= FileReader.getImageArray(grndtruth_masks_file)

            #take only slices in the middle of lungs (1/5 - 4/5)*depth_Z
            begin_slices = full_images_array.shape[0] // 5
            end_slices   = 4 * full_images_array.shape[0] // 5

            PlotsManager.plot_compare_images_masks_allSlices(full_images_array     [begin_slices:end_slices],
                                                             grndtruth_masks_array [begin_slices:end_slices],
                                                             predict_probmaps_array[begin_slices:end_slices],
                                                             isSaveImages=True, outfilespath=SaveImagesPath)

            if (args.saveFeatMapsLayers):
                for ifeatmap in range(num_save_featmaps):
                    ind_featmap = get_index_featmap(ifeatmap)
                    SaveImagesRelPath = tempNameSaveFeatMapsSliceImagesDirs %(filenamenoextension(full_images_file), args.nameSaveModelLayer, ind_featmap+1)
                    SaveImagesPath    = workDirsManager.getNameNewPath(SaveFeatMapsPath, SaveImagesRelPath)

                    PlotsManager.plot_images_masks_allSlices(full_images_array[begin_slices:end_slices],
                                                             predict_featmaps_array[begin_slices:end_slices,...,ifeatmap],
                                                             isSaveImages=True, outfilespath=SaveImagesPath)
                #endfor
    #endfor



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--typedata', default=TYPEDATAPREDICT)
    parser.add_argument('--modelsdir', default='Models_Restart')
    parser.add_argument('--predictionsdir', default='Predictions_NEW')
    parser.add_argument('--lossfun', default=ILOSSFUN)
    parser.add_argument('--listmetrics', type=parseListarg, default=LISTMETRICS)
    parser.add_argument('--prediction_modelFile', default=PREDICTION_MODELFILE)
    parser.add_argument('--predictAccuracyMetrics', default=PREDICTACCURACYMETRICS)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--filterPredictProbMaps', type=str2bool, default=FILTERPREDICTPROBMAPS)
    parser.add_argument('--prop_valid_outUnet', type=float, default=PROP_VALID_OUTUNET)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--extendSizeImages', type=str2bool, default=EXTENDSIZEIMAGES)
    parser.add_argument('--constructInputDataDLCST', type=str2bool, default=CONSTRUCTINPUTDATADLCST)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=False)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=False)
    parser.add_argument('--typeGPUinstalled', type=str, default=TYPEGPUINSTALLED)
    parser.add_argument('--saveFeatMapsLayers', type=str2bool, default=SAVEFEATMAPSLAYERS)
    parser.add_argument('--nameSaveModelLayer', default=NAMESAVEMODELLAYER)
    parser.add_argument('--maxNumSaveFeatMapsLayers', type=int, default=None)
    parser.add_argument('--firstSaveFeatMapsLayers', type=int, default=None)
    parser.add_argument('--savePredictMaskSlices', type=str2bool, default=SAVEPREDICTMASKSLICES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
