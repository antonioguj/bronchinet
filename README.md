# AirwaySegmentationFramework
Framework to do airway segmentation from chest CTs using deep Convolutional Neural Networks.
Implemented in Pytorch, using deep learning libraries i) Keras or ii) Pytorch.

# Dependencies
Need to intall the following python packages (recommended to use virtualenv):
- h5py (>= 2.10.0)
- keras (>= 2.3.1)
- keras-applications (>= 1.0.8)
- keras-preprocessing (>= 1.1.0)
- matplotlib (>= 2.2.4)
- nibabel (>= 2.5.1)
- numpy (>= 1.16.5)
- pydicom (>= 1.3.0)
- pillow (>= 6.2.1)
- scikit-image (>= 0.14.5)
- scikit-learn (>= 0.20.4)
- simpleITX (>= 1.2.4)
- tensorflow (>= 2.0.0)
- tensorflow-gpu (>= 2.0.0)
- torch (>= 1.3.1)

And the following libraries to run deep learning on GPUs:
- cuda: version 9.0. https://developer.nvidia.com/cuda-zone
- cuDNN: https://developer.nvidia.com/rdp/cudnn-download

# Setting up framework
1) Create working directory and set-up the framework:
- mkdir <working_dir> && cd <working_dir>
- ln -s <directory_where_your_data_is> BaseData
- ln -s <directory_where_you_store_this_framework> Code

(The settings for scripts below are input in the file ./Code/Common/Constants.py)
(if needed, indicate "export PYTHONPATH=./Code" in the "~/.bashrc" file)

# Preprocessing data
2) Preprocess data: apply various preprocessing techniques to input data / masks, if needed: rescaling, binarise masks
- python ./Code/Scripts_ImageOperations/ApplyOperationImages.py <dir_input_data> <dir_output_data> --type=[different options]

3) Compute Bounding-boxes of input images:
- python ./Code/Scripts_ImageOperations/ComputeBoundingBoxImages.py

4) Prepare working data: including cropping images / masking ground-truth to lung region...
- python ./Code/Scripts_Experiments/PrepareData.py

5) Distribute data in training / validation / testing:
- python ./Code/Scripts_Experiments/DistributeTrainTestData.py
 
# Training models
6) Train models:
- python ./Code/Scripts_Experiments/TrainingModel.py --modelsdir=<dir_output_models> [if restart model: --cfgfromfile=<file_config_params> --restart_model=True --restart_epoch=<last_epoch_hist_file>]

# Predictions / Postprocessing
7) Compute predictions form trained model:
- python ./Code/Scripts_Experiments/PredictionModel.py <file_trained_model> <dir_output_predictions> --cfgfromfile=<file_config_params>

8) Compute predicted binary masks from probability maps:
- python ./Code/Scripts_ImageOperations/PostprocessPredictions.py <dir_input_pred_probable_maps> <dir_output_pred_binary_masks>

9) Compute predicted centrelines from binary masks:
- python ./Code/Scripts_ImageOperations/ApplyOperationImages.py <dir_input_pred_binary_masks> <dir_output_pred_centrelines> --type=thinning

10) Compute results metrics / accuracy:
- python ./Code/Scripts_ImageOperations/ComputeResultMetrics.py <dir_input_predictions> --inputcentrelinesdir=<dir_input_pred_centrelines>

(7-8-9-10) Do steps 7-8-9-10 at once:)
- python ./Code/Scripts_Experiments/LaunchPredictionsComplete.py <file_trained_model> <dir_output_predictions>
