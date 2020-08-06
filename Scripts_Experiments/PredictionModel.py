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
from dataloaders.dataloader_manager import *
from dataloaders.loadimagedata_manager import *
if TYPE_DNNLIB_USED == 'Keras':
    from networks.keras.metrics import *
    from networks.keras.networks import *
    from networks.keras.visualmodelparams import *
elif TYPE_DNNLIB_USED == 'Pytorch':
    from networks.pytorch.metrics import *
    if ISTESTMODELSWITHGNN:
        from networks.pytorch.gnn_util.NetworksGNNs import *
    else:
        from networks.pytorch.networks import *
    from networks.pytorch.modeltrainer import *
    from networks.pytorch.visualmodelparams import *
from postprocessing.imagereconstructor_manager import *
from preprocessing.imagegenerator_manager import *
from collections import OrderedDict
import argparse


def func_extract_caseprocname_filename(in_filename):
    return basename_file_noext(in_filename).replace('images_proc', '')



def main(args):
    # First thing, set session in the selected(s) devices: CPU or GPU
    #set_session_in_selected_device(use_GPU_device=True,
    #                               type_GPU_installed=TYPEGPUINSTALLED)

    # ---------- SETTINGS ----------
    nameInputImagesFiles      = 'images_proc*.nii.gz'
    nameInputLabelsFiles      = 'labels_proc*.nii.gz'
    nameInputExtraLabelsFiles = 'cenlines_proc*.nii.gz'
    if (args.saveFeatMapsLayers):
        nameOutputPredictionFiles = 'featmaps_proc%s_lay-%s_feat%0.2i.nii.gz'
    else:
        nameOutputPredictionFiles = 'probmaps_proc%s.nii.gz'
    # ---------- SETTINGS ----------



    workDirsManager      = GeneralDirManager(args.basedir)
    TestingDataPath      = workDirsManager.get_pathdir_exist        (args.testdatadir)
    InputReferKeysFile   = workDirsManager.getNameExistBaseDataFile(args.nameInputReferKeysFile)
    OutputPredictionsPath= workDirsManager.get_pathdir_new          (args.nameOutputPredictionsRelPath)
    OutputReferKeysFile  = workDirsManager.get_pathfile_new          (args.nameOutputReferKeysFile)

    listTestImagesFiles  = list_files_dir(TestingDataPath, nameInputImagesFiles)
    in_dictReferenceKeys = read_dictionary(InputReferKeysFile)



    # LOADING MODEL
    # ----------------------------------------------
    print("-" * 30)
    print("Loading saved model...")
    print("-" * 30)

    print("Loading full model: weights, optimizer, loss, metrics ... and restarting...")
    modelSavedPath = args.predsmodelfile
    print("Restarting from file: \'%s\'..." %(modelSavedPath))

    if TYPE_DNNLIB_USED == 'Keras':
        loss_fun = DICTAVAILLOSSFUNS(args.lossfun, is_masks_exclude=args.masksToRegionInterest).lossfun
        metrics = [DICTAVAILMETRICFUNS(imetrics, is_masks_exclude=args.masksToRegionInterest).renamed_compute() for imetrics in args.listmetrics]
        custom_objects = dict(map(lambda fun: (fun.__name__, fun), [loss_fun] + metrics))
        # load and compile model
        model = UNet.get_load_saved_model(modelSavedPath, custom_objects=custom_objects)

        size_output_modelnet = tuple(model.get_size_output()[1:])

        # output model summary
        model.summary()

    elif TYPE_DNNLIB_USED == 'Pytorch':
        # load and compile model
        if args.isModelsWithGNN:
            dict_added_model_input_args = {'isUse_valid_convs': args.isValidConvolutions,
                                           'isGNN_with_attention_lays': args.isGNNwithAttentionLays,
                                           'source_dir_adjs': SOURCEDIR_ADJS}
        else:
            dict_added_model_input_args = {}

        trainer = ModelTrainer.load_model_full(modelSavedPath, dict_added_model_input_args=dict_added_model_input_args)

        size_output_modelnet = tuple(trainer._networks.get_size_output()[1:])

        # output model summary
        trainer.summary_model()

    if args.isValidConvolutions:
        print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..." %(str(args._size_image_in),
                                                                                              str(size_output_modelnet)))

    if (args.saveFeatMapsLayers):
        print("Compute and store model feature maps, from model layer \'%s\'..." %(args.nameSaveModelLayer))
        if TYPE_DNNLIB_USED == 'Keras':
            visual_model_params = VisualModelParams(model, args._size_image_in)
        elif TYPE_DNNLIB_USED == 'Pytorch':
            visual_model_params = VisualModelParams(trainer._networks, args._size_image_in)
    # ----------------------------------------------


    # Create Image generators / Reconstructors
    # ----------------------------------------------
    test_images_generator = get_images_generator(args._size_image_in,
                                                 args.slidingWindowImages,
                                                 args.propOverlapSlidingWindow,
                                                 args.randomCropWindowImages,
                                                 args.numRandomImagesPerVolumeEpoch,
                                                 args.transformationRigidImages,
                                                 args.transformElasticDeformImages)
    images_reconstructor = get_images_reconstructor(args._size_image_in,
                                                    args.slidingWindowImages,
                                                    args.propOverlapSlidingWindow,
                                                    args.randomCropWindowImages,
                                                    args.numRandomImagesPerVolumeEpoch,
                                                    use_transform_rigid_images=False,
                                                    is_output_nnet_validconvs=args.isValidConvolutions,
                                                    size_output_image=size_output_modelnet,
                                                    is_filter_output_nnet=FILTERPREDICTPROBMAPS,
                                                    prop_filter_output_nnet=PROP_VALID_OUTUNET)
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
            in_testXData = LoadImageDataManager.load_1file(in_testXData_file)
            in_testXData_batches = get_batchdata_generator_with_generator(args._size_image_in,
                                                                          [in_testXData],
                                                                          [in_testXData],
                                                                          test_images_generator,
                                                                          batch_size=1,
                                                                          is_output_nnet_validconvs=args.isValidConvolutions,
                                                                          size_output_images=size_output_modelnet,
                                                                          shuffle=False)
        else:
            in_testXData_batches = LoadImageDataInBatchesManager(args._size_image_in).load_1file(in_testXData_file)
            in_testXData_batches = np.expand_dims(in_testXData_batches, axis=0)

        print("Total Data batches generated: %s..." % (len(in_testXData_batches)))
        # -----------------------------------------------------------------------------


        # Compute Model Evaluation
        if (args.saveFeatMapsLayers):
            print("Evaluate model feature maps...")
            out_predict_yData = visual_model_params.get_feature_maps(in_testXData_batches, args.nameSaveModelLayer)
        else:
            print("Evaluate model...")
            if TYPE_DNNLIB_USED == 'Keras':
                out_predict_yData = model.predict(in_testXData_batches.get_full_data(),  batch_size=1)
            elif TYPE_DNNLIB_USED == 'Pytorch':
                out_predict_yData = trainer.predict(in_testXData_batches)


        # -----------------------------------------------------------------------------
        print("Reconstruct Prediction in batches to full size...")
        out_recons_image_shape = ImageFileReader.get_image_size(in_testXData_file) # init with size of "ifile"
        images_reconstructor.update_image_data(out_recons_image_shape)

        out_prediction_array = images_reconstructor.compute(out_predict_yData)
        # -----------------------------------------------------------------------------


        # Output predictions
        in_referkey_file = in_dictReferenceKeys[basename_file_noext(in_testXData_file)]
        in_caseprocname_file = func_extract_caseprocname_filename(in_testXData_file)

        if (args.saveFeatMapsLayers):
            num_featmaps = out_prediction_array.shape[-1]
            print("Output model Feature maps (\'%s\' in total)..." %(num_featmaps))

            for ifeatmap in range(num_featmaps):
                output_pred_file = join_path_names(OutputPredictionsPath, nameOutputPredictionFiles % (in_caseprocname_file, args.nameSaveModelLayer, ifeatmap + 1))
                print("Output: \'%s\', of dims \'%s\'..." %(basename(output_pred_file), out_prediction_array[...,ifeatmap].shape))

                ImageFileReader.write_image(output_pred_file, out_prediction_array[..., ifeatmap])

                # save this prediction in reference keys
                outdict_referenceKeys[basename_file_noext(output_pred_file)] = basename(in_referkey_file)
            #endfor
        else:
            output_pred_file = join_path_names(OutputPredictionsPath, nameOutputPredictionFiles % (in_caseprocname_file))
            print("Output: \'%s\', of dims \'%s\'..." % (basename(output_pred_file), out_prediction_array.shape))

            ImageFileReader.write_image(output_pred_file, out_prediction_array)

            # save this prediction in reference keys
            outdict_referenceKeys[basename_file_noext(output_pred_file)] = basename(in_referkey_file)
    #endfor

    # Save dictionary in file
    save_dictionary(OutputReferKeysFile, outdict_referenceKeys)
    save_dictionary_csv(OutputReferKeysFile.replace('.npy', '.csv'), outdict_referenceKeys)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('predsmodelfile', type=str)
    parser.add_argument('--cfgfromfile', type=str, default=None)
    parser.add_argument('--testdatadir', type=str, default=NAME_TESTINGDATA_RELPATH)
    parser.add_argument('--nameInputReferKeysFile', type=str, default=NAME_REFERKEYSPROCIMAGE_FILE)
    parser.add_argument('--nameOutputPredictionsRelPath', type=str, default=NAME_TEMPOPOSTERIORS_RELPATH)
    parser.add_argument('--nameOutputReferKeysFile', type=str, default=NAME_REFERKEYSPOSTERIORS_FILE)
    parser.add_argument('--size_in_images', type=str2tuple_int, default=IMAGES_DIMS_Z_X_Y)
    parser.add_argument('--isValidConvolutions', type=str2bool, default=ISVALIDCONVOLUTIONS)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=True)
    parser.add_argument('--propOverlapSlidingWindow', type=str2tuple_float, default=PROPOVERLAPSLIDINGWINDOW_TESTING_Z_X_Y)
    parser.add_argument('--randomCropWindowImages', type=str2tuple_float, default=False)
    parser.add_argument('--numRandomImagesPerVolumeEpoch', type=str2tuple_float, default=0)
    parser.add_argument('--transformationRigidImages', type=str2bool, default=False)
    parser.add_argument('--transformElasticDeformImages', type=str2bool, default=False)
    parser.add_argument('--saveFeatMapsLayers', type=str2bool, default=SAVEFEATMAPSLAYERS)
    parser.add_argument('--nameSaveModelLayer', type=str, default=NAMESAVEMODELLAYER)
    parser.add_argument('--lossfun', type=str, default=ILOSSFUN)
    parser.add_argument('--listmetrics', type=str2list_string, default=LISTMETRICS)
    parser.add_argument('--isModelsWithGNN', type=str2bool, default=ISTESTMODELSWITHGNN)
    parser.add_argument('--isGNNwithAttentionLays', type=str2bool, default=ISGNNWITHATTENTIONLAYS)
    args = parser.parse_args()

    if args.cfgfromfile:
        if not is_exist_file(args.cfgfromfile):
            message = "Config params file not found: \'%s\'..." % (args.cfgfromfile)
            catch_error_exception(message)
        else:
            input_args_file = read_dictionary_configparams(args.cfgfromfile)
        print("Set up experiments with parameters from file: \'%s\'" %(args.cfgfromfile))
        #args.basedir               = str(input_args_file['basedir'])
        args.size_in_images        = str2tuple_int(input_args_file['size_in_images'])
        args.masksToRegionInterest = str2bool(input_args_file['masksToRegionInterest'])
        args.isValidConvolutions   = str2bool(input_args_file['isValidConvolutions'])
        args.isGNNwithAttentionLays = str2bool(input_args_file['isGNNwithAttentionLays'])

    print("Print input arguments...")
    for key, value in sorted(vars(args).items()):
        print("\'%s\' = %s" %(key, value))

    main(args)
