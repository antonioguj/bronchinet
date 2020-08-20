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

(The settings for scripts below are input in the file ./Code/common/constants.py)
(if needed, indicate "export PYTHONPATH=./Code" in the "~/.bashrc" file)

# Preprocessing data
2) Preprocess data: apply various preprocessing techniques to input data / masks, if needed: rescaling, binarise masks
- python ./Code/scripts_util/apply_operation_images.py <dir_input_data> <dir_output_data> --type=[different options]

3) Compute Bounding-boxes of input images:
- python ./Code/scripts_airway_segmentation/compute_boundingbox_images.py

4) Prepare working data: including cropping images / masking ground-truth to lung region...
- python ./Code/scripts_experiments/prepare_data.py

5) Distribute data in training / validation / testing:
- python ./Code/scripts_experiments/distribute_data_train_valid_test.py
 
# Training models
6) Train models:
- python ./Code/scripts_experiments/train_model.py --modelsdir=<dir_output_models> [if restart model: --is_config_fromfile=<file_config> --is_restart_model=True]

# Predictions / Postprocessing
7) Compute predictions form trained model:
- python ./Code/scripts_experiments/predict_model.py <file_trained_model> <dir_output_predictions> --is_config_fromfile=<file_config>

8) Compute predicted binary masks from probability maps:
- python ./Code/scripts_airway_segmentation/postprocess_predictions.py <dir_input_pred_probable_maps> <dir_output_pred_binary_masks>

9) Compute predicted centrelines from binary masks:
- python ./Code/scripts_util/apply_operation_images.py <dir_input_pred_binary_masks> <dir_output_pred_centrelines> --type=thinning

10) Compute results metrics / accuracy:
- python ./Code/scripts_airway_segmentation/compute_result_metrics.py <dir_input_predictions> --inputcentrelinesdir=<dir_input_pred_centrelines>

(7-8-9-10) Do steps 7-8-9-10 at once:)
- python ./Code/scripts_airway_segmentation/launch_pipeline_predictions.py <file_trained_model> <dir_output_predictions>