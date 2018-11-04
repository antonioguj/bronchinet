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
from Networks.Metrics import *
from Networks.Networks import *
from Networks.VisualModelParams import *
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

    InputImagesPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputImagesRelPath)
    InputMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, nameInputMasksRelPath )

    listFullImagesFiles     = findFilesDir(InputImagesPath, nameInputImagesFiles)
    listGrndTruthMasksFiles = findFilesDir(InputMasksPath,  nameInputMasksFiles)


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

    train_model_funs = [DICTAVAILLOSSFUNS(args.lossfun)] + [DICTAVAILMETRICFUNS(imetrics, set_fun_name=True) for imetrics in args.listmetrics]
    custom_objects = dict(map(lambda fun: (fun.__name__, fun), train_model_funs))

    model = NeuralNetwork.getLoadSavedModel(modelSavedPath, custom_objects=custom_objects)

    predictAccuracyMetrics_before = args.predictAccuracyMetrics + ('_Masked' if args.masksToRegionInterest else '')

    computePredictAccuracy_before = DICTAVAILMETRICFUNS(predictAccuracyMetrics_before, use_in_Keras=False)

    computePredictAccuracy_after  = DICTAVAILMETRICFUNS(args.predictAccuracyMetrics, use_in_Keras=False)

    if (args.filterPredictProbMaps):
        size_valid_outUnet = tuple([int(args.prop_valid_outUnet * elem) for elem in IMAGES_DIMS_Z_X_Y])

        print("Filter output probability maps. With size of valid convs Unet: %s..." %(str(size_valid_outUnet)))

        filtering_valid_Unet_output = FilteringValidUnetOutput3D(IMAGES_DIMS_Z_X_Y, size_valid_outUnet)

    if (args.saveFeatMapsLayers):
        visual_model_params = VisualModelParams(model, IMAGES_DIMS_Z_X_Y)




    print("-" * 30)
    print("Predicting model...")
    print("-" * 30)

    for i, (test_xData_file, test_yData_file) in enumerate(zip(listTestImagesDataFiles, listTestMasksDataFiles)):

        print('\'%s\'...' % (test_xData_file))

        # Assign original images and masks files
        index_input_images = getIndexOriginImagesFile(basename(test_xData_file), beginString='images', firstIndex='01')

        full_images_file    = listFullImagesFiles    [index_input_images]
        grndtruth_masks_file= listGrndTruthMasksFiles[index_input_images]

        print("assigned to original files: '%s' and '%s'..." %(basename(full_images_file), basename(grndtruth_masks_file)))


        # Loading Data
        print("Loading data...")

        if (args.slidingWindowImages or args.transformationImages):

            test_images_generator = getImagesDataGenerator3D(args.slidingWindowImages,
                                                             args.prop_overlap_Z_X_Y,
                                                             args.transformationImages,
                                                             args.elasticDeformationImages)

            (test_xData, test_yData) = LoadDataManagerInBatches_DataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                              test_images_generator,
                                                                              num_classes_out=num_classes_out).loadData_1File(test_xData_file,
                                                                                                                              test_yData_file,
                                                                                                                              shuffle_images=False)
        else:
            (test_xData, test_yData) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_1File(test_xData_file,
                                                                                                  test_yData_file,
                                                                                                  shuffle_images=False)


        # EVALUATE MODEL
        print("Evaluate model...")
        predict_yData = model.predict(test_xData, batch_size=1)

        if (args.filterPredictProbMaps):
            # Filter predicted probability maps to remove border effects
            size_predict_yData = len(predict_yData)

            for i in range(size_predict_yData):
                predict_yData[i] = filtering_valid_Unet_output.get_filtered_image_array(predict_yData[i])
            #endfor



        # Reconstruct predicted batch images to full 3D array
        fullsize_ydata_shape = FileReader.getImageSize(test_yData_file)

        if (args.slidingWindowImages or args.transformationImages):
            images_reconstructor = getImagesReconstructor3D(args.slidingWindowImages,
                                                            fullsize_ydata_shape,
                                                            args.prop_overlap_Z_X_Y,
                                                            use_TransformationImages=args.transformationImages)
        else:
            images_reconstructor = SlicingReconstructorImages3D(IMAGES_DIMS_Z_X_Y,
                                                                fullsize_ydata_shape)

        predict_probmaps_array = images_reconstructor.compute(predict_yData)



        # Reconstruct predicted cropped probability maps to original size of input masks file
        grndtruth_masks_array_shape = FileReader.getImageSize(grndtruth_masks_file)

        if (args.cropImages):
            if (predict_probmaps_array.shape > grndtruth_masks_array_shape):
                message = "size of predicted probability maps array: %s, cannot be larger than ground-truth masks: %s..." %(predict_probmaps_array.shape, grndtruth_masks_array_shape)
                CatchErrorException(message)
            else:
                print("Predicted probability maps are cropped. "
                      "Extend array size from %s to original %s..."%(predict_probmaps_array.shape, grndtruth_masks_array_shape))

            bounding_box = dict_masks_boundingBoxes[filenamenoextension(full_images_file)]

            predict_probmaps_array = ExtendImages.compute3D(predict_probmaps_array, bounding_box,
                                                            grndtruth_masks_array_shape)

        elif (args.extendSizeImages):
            if (predict_probmaps_array.shape < grndtruth_masks_array_shape):
                message = "size of predicted probability maps array: %s, cannot be smaller than ground-truth masks: %s..." %(predict_probmaps_array.shape, grndtruth_masks_array_shape)
                CatchErrorException(message)
            else:
                print("Predicted probability maps are extended. "
                      "Crop array size from %s to original %s..."%(predict_probmaps_array.shape, grndtruth_masks_array_shape))

            bounding_box = dict_masks_boundingBoxes[filenamenoextension(full_images_file)]

            predict_probmaps_array = CropImages.compute3D(predict_probmaps_array, bounding_box)

        else:
            if (predict_probmaps_array.shape != grndtruth_masks_array_shape):
                message = "size of predicted probability maps array: %s, not equal to size of ground-truth masks: %s..." %(predict_probmaps_array.shape, grndtruth_masks_array_shape)
                CatchErrorException(message)



        if (args.masksToRegionInterest):
            print("Mask predicted probability maps to Region of Interest: lungs...")

            lungs_masks_file = listLungsMasksFiles[index_input_images]

            print("assigned to: '%s'..." %(basename(lungs_masks_file)))

            lungs_masks_array = FileReader.getImageArray(lungs_masks_file)

            predict_probmaps_array = OperationsBinaryMasks.reverse_mask_exclude_voxels_fillzero(predict_probmaps_array,
                                                                                                lungs_masks_array)


        # Compute test accuracy
        accuracy_before = computePredictAccuracy_before(test_yData.astype(FORMATPROBABILITYDATA),
                                                        predict_yData)

        grndtruth_masks_array = FileReader.getImageArray(grndtruth_masks_file)

        accuracy_after  = computePredictAccuracy_after(grndtruth_masks_array,
                                                       predict_probmaps_array)

        print("Computed accuracy (before post-processing): %s..." %(accuracy_before))
        print("Computed accuracy (after post-processing): %s..." %(accuracy_after))



        # Save reconstructed predict probability maps (or thresholding masks)
        print("Saving predict probability maps, with dims: %s..." %(tuple2str(predict_probmaps_array.shape)))

        out_predictMasksFilename = tempNamePredictMasksFiles %(filenamenoextension(full_images_file))
        out_predictMasksFilename = joinpathnames(PredictDataPath, out_predictMasksFilename)

        FileReader.writeImageArray(out_predictMasksFilename, predict_probmaps_array)


        # Save predictions in images
        if (args.savePredictMaskSlices):
            SaveImagesRelPath = tempNameSavePredictMaskSliceImagesDirs %(filenamenoextension(full_images_file))
            SaveImagesPath    = workDirsManager.getNameNewPath(PredictDataPath, SaveImagesRelPath)

            full_images_array = FileReader.getImageArray(full_images_file)

            #take only slices in the middle of lungs (1/5 - 4/5)*depth_Z
            begin_slices = full_images_array.shape[0] // 5
            end_slices   = 4 * full_images_array.shape[0] // 5

            PlotsManager.plot_compare_images_masks_allSlices(full_images_array     [begin_slices:end_slices],
                                                             grndtruth_masks_array [begin_slices:end_slices],
                                                             predict_probmaps_array[begin_slices:end_slices],
                                                             isSaveImages=True, outfilespath=SaveImagesPath)



        # *************** VISUALIZE FEATURE MAPS ***************
        if (args.saveFeatMapsLayers):
            # visualize feature maps evaluated at test image
            print("Compute feature maps of evaluated model...")

            predict_featmaps_data = visual_model_params.get_feature_maps(test_xData, args.nameSaveModelLayer)

            num_save_featmaps = predict_featmaps_data.shape[-1]

            fullsize_featmaps_shape = [num_save_featmaps] + list(fullsize_ydata_shape)

            predict_featmaps_array = np.zeros(fullsize_featmaps_shape, dtype=FORMATFEATUREDATA)

            for ifeatmap in range(num_save_featmaps):
                in_predict_ifeatmap_data = np.expand_dims(predict_featmaps_data[..., ifeatmap], axis=-1)
                predict_featmaps_array[ifeatmap] = images_reconstructor.compute(in_predict_ifeatmap_data)
            # endfor



            # Reconstruct predicted cropped probability maps to original size of input masks file
            if (args.cropImages or args.extendSizeImages):

                new_predict_featmaps_array_shape = [num_save_featmaps] + list(grndtruth_masks_array_shape)

                new_predict_featmaps_array = np.zeros(new_predict_featmaps_array_shape, dtype=FORMATFEATUREDATA)

                bounding_box = dict_masks_boundingBoxes[filenamenoextension(full_images_file)]

                if (args.cropImages):
                    print("Predicted feature maps are cropped. "
                          "Extend array size from %s to original %s..." %(predict_featmaps_array.shape, new_predict_featmaps_array_shape))

                    for ifeatmap in range(num_save_featmaps):
                        new_predict_featmaps_array[ifeatmap] = ExtendImages.compute3D(predict_featmaps_array[ifeatmap],
                                                                                      bounding_box, grndtruth_masks_array_shape)
                    #endfor

                elif (args.extendSizeImages):
                    print("Predicted probability maps are extended. "
                          "Crop array size from %s to original %s..." %(predict_featmaps_array.shape, new_predict_featmaps_array_shape))

                    for ifeatmap in range(num_save_featmaps):
                        new_predict_featmaps_array[ifeatmap] = CropImages.compute3D(predict_featmaps_array[ifeatmap],
                                                                                    bounding_box)
                    #endfor

                predict_featmaps_array = new_predict_featmaps_array


            if (args.masksToRegionInterest):
                print("Mask predicted feature maps to Region of Interest: lungs...")

                for ifeatmap in range(num_save_featmaps):
                    predict_featmaps_array[ifeatmap] = OperationsBinaryMasks.reverse_mask_exclude_voxels_fillzero(predict_featmaps_array[ifeatmap],
                                                                                                                  lungs_masks_array)
                #endfor


            # Save feature maps computed on evaluated model
            if (args.saveFeatMapsLayers):
                print("Saving computed feature maps, with dims: %s..." % (tuple2str(predict_featmaps_array.shape)))

                SaveFeatMapsRelPath = tempNameSaveFeatMapsDirs %(filenamenoextension(full_images_file), args.nameSaveModelLayer)
                SaveFeatMapsPath    = workDirsManager.getNameNewPath(PredictDataPath, SaveFeatMapsRelPath)

                for ifeatmap in range(num_save_featmaps):
                    out_featMapsFilename = tempNameSaveFeatMapsFiles %(filenamenoextension(full_images_file), args.nameSaveModelLayer, ifeatmap+1)
                    out_featMapsFilename = joinpathnames(SaveFeatMapsPath, out_featMapsFilename)

                    FileReader.writeImageArray(out_featMapsFilename, predict_featmaps_array[ifeatmap])
                # endfor


                # Save feature maps in images
                if (args.savePredictMaskSlices):
                    for ifeatmap in range(num_save_featmaps):
                        SaveImagesRelPath = tempNameSaveFeatMapsSliceImagesDirs %(filenamenoextension(full_images_file), args.nameSaveModelLayer, ifeatmap+1)
                        SaveImagesPath    = workDirsManager.getNameNewPath(SaveFeatMapsPath, SaveImagesRelPath)

                        PlotsManager.plot_images_masks_allSlices(full_images_array[begin_slices:end_slices],
                                                                 predict_featmaps_array[ifeatmap][begin_slices:end_slices],
                                                                 isSaveImages=True, outfilespath=SaveImagesPath)
                    #endfor
        # *************** VISUALIZE FEATURE MAPS ***************
    #endfor



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--typedata', default=TYPEDATAPREDICT)
    parser.add_argument('--modelsdir', default='Models_Restart')
    parser.add_argument('--predictionsdir', default='Predictions_NEW')
    parser.add_argument('--lossfun', default=ILOSSFUN)
    parser.add_argument('--listmetrics', type=parseListarg, default=LISTMETRICS)
    parser.add_argument('--num_featmaps_firstlayer', type=int, default=NUM_FEATMAPS_FIRSTLAYER)
    parser.add_argument('--prediction_modelFile', default=PREDICTION_MODELFILE)
    parser.add_argument('--predictAccuracyMetrics', default=PREDICTACCURACYMETRICS)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--filterPredictProbMaps', type=str2bool, default=FILTERPREDICTPROBMAPS)
    parser.add_argument('--prop_valid_outUnet', type=float, default=PROP_VALID_OUTUNET)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--extendSizeImages', type=str2bool, default=EXTENDSIZEIMAGES)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=False)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=False)
    parser.add_argument('--typeGPUinstalled', type=str, default=TYPEGPUINSTALLED)
    parser.add_argument('--saveFeatMapsLayers', type=str2bool, default=SAVEFEATMAPSLAYERS)
    parser.add_argument('--nameSaveModelLayer', default=NAMESAVEMODELLAYER)
    parser.add_argument('--savePredictMaskSlices', type=str2bool, default=SAVEPREDICTMASKSLICES)
    args = parser.parse_args()

    if (args.masksToRegionInterest):
        args.lossfun     = args.lossfun + '_Masked'
        args.listmetrics = [item + '_Masked' for item in args.listmetrics]

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
