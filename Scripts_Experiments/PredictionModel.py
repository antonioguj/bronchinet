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
    from Networks_Keras.Metrics import *
    from Networks_Keras.Networks import *
    from Networks_Keras.VisualModelParams import *
elif TYPE_DNNLIBRARY_USED == 'Pytorch':
    from DataLoaders.BatchDataGenerator_Pytorch import *
    from Networks_Pytorch.Metrics import *
    if ISTESTMODELSWITHGNN:
        from Networks_Pytorch.NetworksGNNs import *
    else:
        from Networks_Pytorch.Networks import *
    from Networks_Pytorch.Trainers import *
    from Networks_Pytorch.VisualModelParams import *
from Postprocessing.ImageReconstructorManager import *
from Preprocessing.ImageGeneratorManager import *
from Preprocessing.OperationImages import *
from Preprocessing.OperationMasks import *
import argparse



def main(args):
    # First thing, set session in the selected(s) devices: CPU or GPU
    set_session_in_selected_device(use_GPU_device=True,
                                   type_GPU_installed=TYPEGPUINSTALLED)

    # ---------- SETTINGS ----------
    nameInputRoiMasksRelPath   = 'Lungs_Proc/'
    nameInputReferFilesRelPath = 'Images_Proc/'
    nameImagesFiles            = 'images*' + getFileExtension(FORMATTRAINDATA)
    nameLabelsFiles            = 'labels*' + getFileExtension(FORMATTRAINDATA)
    nameInputRoiMasksFiles     = '*_lungs.nii.gz'
    nameInputReferFiles        = '*.nii.gz'
    nameCropBoundingBoxes      = 'cropBoundingBoxes_images.npy'
    #nameRescaleFactors         = 'rescaleFactors_images.npy'
    nameOutputPredictionsRelPath = args.predictionsdir

    if (args.saveFeatMapsLayers):
        nameOutputPredictionFiles = '%s_featmap_lay-%s_feat%0.2i.nii.gz'
    else:
        nameOutputPredictionFiles  = '%s_probmap.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager      = WorkDirsManager(args.basedir)
    TestingDataPath      = workDirsManager.getNameExistPath        (args.testdatadir)
    InputReferFilesPath  = workDirsManager.getNameExistBaseDataPath(nameInputReferFilesRelPath)
    OutputPredictionsPath= workDirsManager.getNameNewPath          (nameOutputPredictionsRelPath)

    listTestImagesFiles = findFilesDirAndCheck(TestingDataPath,     nameImagesFiles)
    listTestLabelsFiles = findFilesDirAndCheck(TestingDataPath,     nameLabelsFiles)
    listInputReferFiles = findFilesDirAndCheck(InputReferFilesPath, nameInputReferFiles)

    if (args.masksToRegionInterest):
        InputRoiMasksPath      = workDirsManager.getNameExistBaseDataPath(nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath, nameInputRoiMasksFiles)

    if (args.cropImages):
        cropBoundingBoxesFileName = joinpathnames(workDirsManager.getNameBaseDataPath(), nameCropBoundingBoxes)
        dict_cropBoundingBoxes = readDictionary(cropBoundingBoxesFileName)

    #if (args.rescaleImages):
    #    rescaleFactorsFileName = joinpathnames(workDirsManager.getNameBaseDataPath(), nameRescaleFactors)
    #    dict_rescaleFactors = readDictionary(rescaleFactorsFileName)



    # LOADING MODEL
    # ----------------------------------------------
    print("-" * 30)
    print("Loading saved model...")
    print("-" * 30)

    print("Loading full model: weights, optimizer, loss, metrics ... and restarting...")
    modelSavedPath = args.predsmodelfile
    print("Restarting from file: \'%s\'..." %(modelSavedPath))

    if TYPE_DNNLIBRARY_USED == 'Keras':
        loss_fun = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest).loss
        metrics = [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest).get_renamed_compute() for imetrics in args.listmetrics]
        custom_objects = dict(map(lambda fun: (fun.__name__, fun), [loss_fun] + metrics))
        # load and compile model
        model = NeuralNetwork.get_load_saved_model(modelSavedPath, custom_objects=custom_objects)

        # output model summary
        model.summary()

    elif TYPE_DNNLIBRARY_USED == 'Pytorch':
        # load and compile model
        if args.isModelsWithGNN:
            dict_added_model_input_args = {'isUse_valid_convs': args.isValidConvolutions,
                                           'isGNN_with_attention_lays': args.isGNNwithAttentionLays,
                                           'source_dir_adjs': SOURCEDIR_ADJS}
        else:
            dict_added_model_input_args = {}

        trainer = Trainer.load_model_full(modelSavedPath,
                                          dict_added_model_input_args=dict_added_model_input_args)

        size_output_modelnet = tuple(trainer.model_net.get_size_output()[1:])
        if args.isValidConvolutions:
            print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..." %(str(args.size_in_images),
                                                                                                  str(size_output_modelnet)))
        # output model summary
        #trainer.get_summary_model()

    if (args.saveFeatMapsLayers):
        print("Compute and store model feature maps, from model layer \'%s\'..." %(args.nameSaveModelLayer))
        if TYPE_DNNLIBRARY_USED == 'Keras':
            visual_model_params = VisualModelParams(model, args.size_in_images)
        elif TYPE_DNNLIBRARY_USED == 'Pytorch':
            visual_model_params = VisualModelParams(trainer.model_net, args.size_in_images)


    test_images_generator = getImagesDataGenerator3D(args.size_in_images,
                                                     args.slidingWindowImages,
                                                     args.slidewin_propOverlap,
                                                     args.transformationImages,
                                                     args.elasticDeformationImages)
    images_reconstructor = getImagesReconstructor3D(args.size_in_images,
                                                    args.slidingWindowImages,
                                                    args.slidewin_propOverlap,
                                                    use_TransformationImages=False,
                                                    isUse_valid_convs=args.isValidConvolutions,
                                                    size_output_model=size_output_modelnet,
                                                    isfilter_valid_outUnet=FILTERPREDICTPROBMAPS,
                                                    prop_valid_outUnet=PROP_VALID_OUTUNET)
    # ----------------------------------------------



    # START ANALYSIS
    # ----------------------------------------------
    print("-" * 30)
    print("Predicting model...")
    print("-" * 30)

    for ifile, in_testXData_file in enumerate(listTestImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_testXData_file)))

        # COMPUTE PREDICTION
        # ------------------------------------------
        print("Loading data...")
        if (args.slidingWindowImages or args.transformationImages):
            if TYPE_DNNLIBRARY_USED == 'Keras':
                in_testXData_batches = LoadDataManagerInBatches_DataGenerator(args.size_in_images,
                                                                              test_images_generator).loadData_1File(in_testXData_file,
                                                                                                                    shuffle_images=False)
            elif TYPE_DNNLIBRARY_USED == 'Pytorch':
                in_testXData = LoadDataManager.loadData_1File(in_testXData_file)
                in_testXData_batches = TrainingBatchDataGenerator(args.size_in_images,
                                                                  [in_testXData],
                                                                  [in_testXData],
                                                                  test_images_generator,
                                                                  batch_size=1,
                                                                  isUse_valid_convs=args.isValidConvolutions,
                                                                  size_output_model=size_output_modelnet,
                                                                  shuffle=False)
        else:
            in_testXData_batches = LoadDataManagerInBatches(args.size_in_images).loadData_1File(in_testXData_file)
            in_testXData_batches = np.expand_dims(in_testXData_batches, axis=0)

        print("Total Data batches generated: %s..." % (len(in_testXData_batches)))

        if (args.saveFeatMapsLayers):
            print("Evaluate model feature maps...")
            out_predict_yData = visual_model_params.get_feature_maps(in_testXData_batches, args.nameSaveModelLayer)
        else:
            print("Evaluate model...")
            if TYPE_DNNLIBRARY_USED == 'Keras':
                out_predict_yData = model.predict(in_testXData_batches, batch_size=1)
            elif TYPE_DNNLIBRARY_USED == 'Pytorch':
                out_predict_yData = trainer.predict(in_testXData_batches)
        # ------------------------------------------



        # RECONSTRUCT FULL-SIZE PREDICTION
        # ------------------------------------------
        print("Reconstruct prediction to full size...")
        # Assign original images and masks files
        index_referimg_file = getIndexOriginImagesFile(basename(in_testXData_file), beginString='images', firstIndex='01')
        in_referimage_file = listInputReferFiles[index_referimg_file]
        print("Reference image file: \'%s\'..." %(basename(in_referimage_file)))

        # init reconstructor with size of "ifile"
        out_reconstructImage_shape = FileReader.getImageSize(in_testXData_file)
        images_reconstructor.complete_init_data(out_reconstructImage_shape)

        out_prediction_array = images_reconstructor.compute(out_predict_yData)


        # reconstruct from cropped / rescaled images
        out_fullimage_shape = FileReader.getImageSize(in_referimage_file)
        if (args.saveFeatMapsLayers):
            num_featmaps = out_prediction_array.shape[-1]
            out_fullimage_shape = list(out_fullimage_shape) + [num_featmaps]


        if (args.cropImages):
            crop_bounding_box = dict_cropBoundingBoxes[filenamenoextension(in_referimage_file)]
            print("Predicted data are cropped. Extend array size to original. Bounding-box: \'%s\'..." %(str(crop_bounding_box)))

            out_prediction_array = ExtendImages.compute3D(out_prediction_array, crop_bounding_box, out_fullimage_shape)
            print("Final dims: %s..." %(str(out_prediction_array.shape)))


        if (args.masksToRegionInterest):
            print("Mask predictions to RoI: lungs...")
            in_roimask_file = listInputRoiMasksFiles[index_referimg_file]
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array = FileReader.getImageArray(in_roimask_file)
            out_prediction_array = OperationBinaryMasks.reverse_mask_exclude_voxels_fillzero(out_prediction_array, in_roimask_array)
        # ------------------------------------------



        if (args.saveFeatMapsLayers):
            num_featmaps = out_prediction_array.shape[-1]
            print("Output model feature maps (\'%s\' in total)..." %(num_featmaps))
            for ifeatmap in range(num_featmaps):
                out_prediction_file = nameOutputPredictionFiles %(filenamenoextension(in_referimage_file), args.nameSaveModelLayer, ifeatmap+1)
                out_prediction_file = joinpathnames(OutputPredictionsPath, out_prediction_file)
                print("Output: \'%s\', of dims \'%s\'..." %(basename(out_prediction_file), out_prediction_array[...,ifeatmap].shape))

                FileReader.writeImageArray(out_prediction_file, out_prediction_array[...,ifeatmap])
            #endfor
        else:
            out_prediction_file = joinpathnames(OutputPredictionsPath, nameOutputPredictionFiles %(filenamenoextension(in_referimage_file)))
            print("Output: \'%s\', of dims \'%s\'..." % (basename(out_prediction_file), out_prediction_array.shape))

            FileReader.writeImageArray(out_prediction_file, out_prediction_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('predsmodelfile', type=str, default='Models_Restart')
    parser.add_argument('predictionsdir', type=str, default='Predictions_NEW')
    parser.add_argument('--testdatadir', type=str, default='TestingData')
    parser.add_argument('--cfgfromfile', type=str, default=None)
    parser.add_argument('--size_in_images', type=str2tupleint, default=IMAGES_DIMS_Z_X_Y)
    parser.add_argument('--isValidConvolutions', type=str2bool, default=ISVALIDCONVOLUTIONS)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--extendSizeImages', type=str2bool, default=EXTENDSIZEIMAGES)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--slidewin_propOverlap', type=str2tuplefloat, default=SLIDEWIN_PROPOVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=False)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=False)
    parser.add_argument('--saveFeatMapsLayers', type=str2bool, default=SAVEFEATMAPSLAYERS)
    parser.add_argument('--nameSaveModelLayer', type=str, default=NAMESAVEMODELLAYER)
    parser.add_argument('--lossfun', type=str, default=ILOSSFUN)
    parser.add_argument('--listmetrics', type=parseListarg, default=LISTMETRICS)
    parser.add_argument('--isModelsWithGNN', type=str2bool, default=ISTESTMODELSWITHGNN)
    parser.add_argument('--isGNNwithAttentionLays', type=str2bool, default=ISGNNWITHATTENTIONLAYS)
    args = parser.parse_args()

    if args.cfgfromfile:
        if not isExistfile(args.cfgfromfile):
            message = "Config params file not found: \'%s\'..." % (args.cfgfromfile)
            CatchErrorException(message)
        else:
            input_args_file = readDictionary_configParams(args.cfgfromfile)
        print("Set up experiments with parameters from file: \'%s\'" %(args.cfgfromfile))
        args.size_in_images         = str2tupleint(input_args_file['size_in_images'])
        args.masksToRegionInterest  = str2bool(input_args_file['masksToRegionInterest'])
        args.isValidConvolutions    = str2bool((input_args_file['isValidConvolutions']))
        args.slidingWindowImages    = str2bool((input_args_file['slidingWindowImages']))
        args.slidewin_propOverlap   = str2tuplefloat((input_args_file['slidewin_propOverlap']))
        args.transformationImages   = False   #str2bool((input_args_file['transformationImages']))
        args.elasticDeformationImages= str2bool((input_args_file['elasticDeformationImages']))
        args.isGNNwithAttentionLays = str2bool((input_args_file['isGNNwithAttentionLays']))

    print("Print input arguments...")
    for key, value in sorted(vars(args).iteritems()):
        print("\'%s\' = %s" %(key, value))

    main(args)
