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
from DataLoaders.BatchDataGeneratorManager import *
from DataLoaders.LoadDataManager import *
if TYPE_DNNLIBRARY_USED == 'Keras':
    from Networks_Keras.Metrics import *
    from Networks_Keras.Networks import *
    from Networks_Keras.VisualModelParams import *
elif TYPE_DNNLIBRARY_USED == 'Pytorch':
    from Networks_Pytorch.Metrics import *
    if ISTESTMODELSWITHGNN:
        from Networks_Pytorch.NetworksGNNs import *
    else:
        from Networks_Pytorch.Networks import *
    from Networks_Pytorch.Trainers import *
    from Networks_Pytorch.VisualModelParams import *
from Postprocessing.ImageReconstructorManager import *
from Preprocessing.ImageGeneratorManager import *
from collections import OrderedDict
import argparse


def func_extract_caseprocname_filename(in_filename):
    return basenameNoextension(in_filename).replace('images_proc','')



def main(args):
    # First thing, set session in the selected(s) devices: CPU or GPU
    set_session_in_selected_device(use_GPU_device=True,
                                   type_GPU_installed=args.typeGPUinstalled)

    # ---------- SETTINGS ----------
    nameInputImagesFiles      = 'images_proc*.nii.gz'
    nameInputLabelsFiles      = 'labels_proc*.nii.gz'
    nameInputExtraLabelsFiles = 'cenlines_proc*.nii.gz'
    if (args.saveFeatMapsLayers):
        nameOutputPredictionFiles = 'featmaps_proc%s_lay-%s_feat%0.2i.nii.gz'
    else:
        nameOutputPredictionFiles = 'probmaps_proc%s.nii.gz'
    # ---------- SETTINGS ----------



    workDirsManager      = WorkDirsManager(args.basedir)
    TestingDataPath      = workDirsManager.getNameExistPath        (args.testdatadir)
    InputReferKeysFile   = workDirsManager.getNameExistBaseDataFile(args.nameInputReferKeysFile)
    OutputPredictionsPath= workDirsManager.getNameNewPath          (args.nameOutputPredictionsRelPath)
    OutputReferKeysFile  = workDirsManager.getNameNewFile          (args.nameOutputReferKeysFile)

    listTestImagesFiles   = findFilesDirAndCheck(TestingDataPath, nameInputImagesFiles)
    in_dictReferenceKeys  = readDictionary(InputReferKeysFile)
    prefixPatternInputFiles = getFilePrefixPattern(in_dictReferenceKeys.values()[0])



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

        size_output_modelnet = tuple(model.get_size_output()[1:])

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

        trainer = Trainer.load_model_full(modelSavedPath, dict_added_model_input_args=dict_added_model_input_args)

        size_output_modelnet = tuple(trainer.model_net.get_size_output()[1:])

        # output model summary
        trainer.get_summary_model()

    if args.isValidConvolutions:
        print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..." %(str(args.size_in_images),
                                                                                              str(size_output_modelnet)))

    if (args.saveFeatMapsLayers):
        print("Compute and store model feature maps, from model layer \'%s\'..." %(args.nameSaveModelLayer))
        if TYPE_DNNLIBRARY_USED == 'Keras':
            visual_model_params = VisualModelParams(model, args.size_in_images)
        elif TYPE_DNNLIBRARY_USED == 'Pytorch':
            visual_model_params = VisualModelParams(trainer.model_net, args.size_in_images)
    # ----------------------------------------------


    # Create Image generators / Reconstructors
    # ----------------------------------------------
    test_images_generator = getImagesDataGenerator(args.size_in_images,
                                                   args.slidingWindowImages,
                                                   args.propOverlapSlidingWindow,
                                                   args.randomCropWindowImages,
                                                   args.numRandomImagesPerVolumeEpoch,
                                                   args.transformationRigidImages,
                                                   args.transformElasticDeformImages)
    images_reconstructor = getImagesReconstructor(args.size_in_images,
                                                  args.slidingWindowImages,
                                                  args.propOverlapSlidingWindow,
                                                  args.randomCropWindowImages,
                                                  args.numRandomImagesPerVolumeEpoch,
                                                  use_transformationRigidImages=False,
                                                  is_outputUnet_validconvs=args.isValidConvolutions,
                                                  size_output_images=size_output_modelnet,
                                                  is_filter_output_unet=FILTERPREDICTPROBMAPS,
                                                  prop_filter_output_unet=PROP_VALID_OUTUNET)
    # ----------------------------------------------



    print("-" * 30)
    print("Predicting model...")
    print("-" * 30)

    outdict_referenceKeys = OrderedDict()

    for ifile, in_testXData_file in enumerate(listTestImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_testXData_file)))

        # -----------------------------------------------------------------------------
        print("Loading data...")
        if (args.slidingWindowImages or args.transformationRigidImages):
            in_testXData = LoadDataManager.load_1file(in_testXData_file)
            in_testXData_batches = getBatchDataGeneratorWithGenerator(args.size_in_images,
                                                                      [in_testXData],
                                                                      [in_testXData],
                                                                      test_images_generator,
                                                                      batch_size=1,
                                                                      is_outputUnet_validconvs=args.isValidConvolutions,
                                                                      size_output_images=size_output_modelnet,
                                                                      shuffle=False)
        else:
            in_testXData_batches = LoadDataManagerInBatches(args.size_in_images).load_1file(in_testXData_file)
            in_testXData_batches = np.expand_dims(in_testXData_batches, axis=0)

        print("Total Data batches generated: %s..." % (len(in_testXData_batches)))
        # -----------------------------------------------------------------------------


        # Compute Model Evaluation
        if (args.saveFeatMapsLayers):
            print("Evaluate model feature maps...")
            out_predict_yData = visual_model_params.get_feature_maps(in_testXData_batches, args.nameSaveModelLayer)
        else:
            print("Evaluate model...")
            if TYPE_DNNLIBRARY_USED == 'Keras':
                out_predict_yData = model.predict(in_testXData_batches.get_full_data(),  batch_size=1)
            elif TYPE_DNNLIBRARY_USED == 'Pytorch':
                out_predict_yData = trainer.predict(in_testXData_batches)


        # -----------------------------------------------------------------------------
        print("Reconstruct Prediction in batches to full size...")
        out_recons_image_shape = FileReader.get_image_size(in_testXData_file) # init with size of "ifile"
        images_reconstructor.update_image_data(out_recons_image_shape)

        out_prediction_array = images_reconstructor.compute(out_predict_yData)
        # -----------------------------------------------------------------------------


        # Output predictions
        in_referkey_file = in_dictReferenceKeys[basenameNoextension(in_testXData_file)]
        in_caseprocname_file = func_extract_caseprocname_filename(in_testXData_file)

        if (args.saveFeatMapsLayers):
            num_featmaps = out_prediction_array.shape[-1]
            print("Output model Feature maps (\'%s\' in total)..." %(num_featmaps))

            for ifeatmap in range(num_featmaps):
                output_pred_file = joinpathnames(OutputPredictionsPath, nameOutputPredictionFiles %(in_caseprocname_file, args.nameSaveModelLayer, ifeatmap+1))
                print("Output: \'%s\', of dims \'%s\'..." %(basename(output_pred_file), out_prediction_array[...,ifeatmap].shape))

                FileReader.write_image_array(output_pred_file, out_prediction_array[..., ifeatmap])

                # save this prediction in reference keys
                outdict_referenceKeys[basenameNoextension(output_pred_file)] = basename(in_referkey_file)
            #endfor
        else:
            output_pred_file = joinpathnames(OutputPredictionsPath, nameOutputPredictionFiles % (in_caseprocname_file))
            print("Output: \'%s\', of dims \'%s\'..." % (basename(output_pred_file), out_prediction_array.shape))

            FileReader.write_image_array(output_pred_file, out_prediction_array)

            # save this prediction in reference keys
            outdict_referenceKeys[basenameNoextension(output_pred_file)] = basename(in_referkey_file)
    #endfor

    # Save dictionary in file
    saveDictionary(OutputReferKeysFile, outdict_referenceKeys)
    saveDictionary_csv(OutputReferKeysFile.replace('.npy', '.csv'), outdict_referenceKeys)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('predsmodelfile', type=str)
    parser.add_argument('--cfgfromfile', type=str, default=None)
    parser.add_argument('--testdatadir', type=str, default=NAME_TESTINGDATA_RELPATH)
    parser.add_argument('--nameInputReferKeysFile', type=str, default=NAME_REFERKEYSPROCIMAGE_FILE)
    parser.add_argument('--nameOutputPredictionsRelPath', type=str, default=NAME_TEMPOPOSTERIORS_RELPATH)
    parser.add_argument('--nameOutputReferKeysFile', type=str, default=NAME_REFERKEYSPOSTERIORS_FILE)
    parser.add_argument('--size_in_images', type=str2tupleint, default=IMAGES_DIMS_Z_X_Y)
    parser.add_argument('--isValidConvolutions', type=str2bool, default=ISVALIDCONVOLUTIONS)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=True)
    parser.add_argument('--propOverlapSlidingWindow', type=str2tuplefloat, default=PROPOVERLAPSLIDINGWINDOW_TESTING_Z_X_Y)
    parser.add_argument('--randomCropWindowImages', type=str2tuplefloat, default=False)
    parser.add_argument('--numRandomImagesPerVolumeEpoch', type=str2tuplefloat, default=0)
    parser.add_argument('--transformationRigidImages', type=str2bool, default=False)
    parser.add_argument('--transformElasticDeformImages', type=str2bool, default=False)
    parser.add_argument('--saveFeatMapsLayers', type=str2bool, default=SAVEFEATMAPSLAYERS)
    parser.add_argument('--nameSaveModelLayer', type=str, default=NAMESAVEMODELLAYER)
    parser.add_argument('--lossfun', type=str, default=ILOSSFUN)
    parser.add_argument('--listmetrics', type=parseListarg, default=LISTMETRICS)
    parser.add_argument('--isModelsWithGNN', type=str2bool, default=ISTESTMODELSWITHGNN)
    parser.add_argument('--isGNNwithAttentionLays', type=str2bool, default=ISGNNWITHATTENTIONLAYS)
    parser.add_argument('--typeGPUinstalled', type=str, default=TYPEGPUINSTALLED)
    args = parser.parse_args()

    if args.cfgfromfile:
        if not isExistfile(args.cfgfromfile):
            message = "Config params file not found: \'%s\'..." % (args.cfgfromfile)
            CatchErrorException(message)
        else:
            input_args_file = readDictionary_configParams(args.cfgfromfile)
        print("Set up experiments with parameters from file: \'%s\'" %(args.cfgfromfile))
        args.basedir               = str(input_args_file['basedir'])
        args.size_in_images        = str2tupleint(input_args_file['size_in_images'])
        args.masksToRegionInterest = str2bool(input_args_file['masksToRegionInterest'])
        args.isValidConvolutions   = str2bool(input_args_file['isValidConvolutions'])
        args.isGNNwithAttentionLays = str2bool(input_args_file['isGNNwithAttentionLays'])

    print("Print input arguments...")
    for key, value in sorted(vars(args).iteritems()):
        print("\'%s\' = %s" %(key, value))

    main(args)
