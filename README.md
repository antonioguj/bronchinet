# AirwaySegmentation framework
Code for airway segmentation from chest CTs using deep Convolutional Neural Networks.
Implemented in python3, and both Pytorch and Keras.

# Dependencies
Need to intall these python packages (recommended to use virtualenv):
- elasticdeform (>= 0.4.6) (from https://github.com/gvtulder/elasticdeform/)
- h5py (>= 2.10.0)
- keras (>= 2.3.1)
- keras-applications (>= 1.0.8)
- keras-preprocessing (>= 1.1.2)
- matplotlib (>= 3.2.1)
- nibabel (>= 3.1.0)
- numpy (>= 1.18.4)
- pandas (>= 1.0.3)
- pillow (>= 7.1.2)
- pydicom (>= 1.4.2)
- scikit-image (>= 0.17.2)
- scikit-learn (>= 0.23.1)
- scipy (>= 1.4.1)
- seaborn (>= 0.10.1)
- simpleITX (>= 1.2.4)
- tensorflow (>= 2.1.0)
- tensorflow-gpu (>= 2.1.0)
- tensorflow-tensorboard (>= 1.5.1)
- torch (>= 1.5.0)
- torchsummary (>= 1.5.1)
- torchvision (>= 0.6.0)
- tqdm (>= 4.46.0)

And these libraries to run deep learning on GPUs:
- cuda (>= 10.2): https://developer.nvidia.com/cuda-zone
- cuDNN (>= 7.6.5): https://developer.nvidia.com/rdp/cudnn-download

# Setting up framework
Create working directory and set up the framework:
- mkdir <working_dir> && cd <working_dir>
- ln -s <dir_where_data_stored> BaseData
- ln -s <dir_where_this_framework> Code

(All settings for the scripts below are in the file ./Code/common/constants.py)
[IF NEEDED] (indicate "export PYTHONPATH=./Code" in the "~/.bashrc" file)

# Preprocessing data
1) [IF NEEDED] Preprocess data: apply various operations to input images / masks: rescaling, binarise masks
- python ./Code/scripts_util/apply_operation_images.py <dir_input_data> <dir_output_data> --type=[various option]

2) [IF NEEDED] Compute bounding-boxes around lung masks, for input images:
- python ./Code/scripts_prepdata/compute_boundingbox_images.py --datadir=[path_dir_dataset]

3) Prepare data: include i) crop images, ii) mask ground-truth to lung regions, iii) rescale images.
- python ./Code/scripts_prepdata/prepare_data.py --datadir=[path_dir_dataset]

# Training models
1) Distribute data in training / validation / testing:
- python ./Code/scripts_experiments/distribute_data.py --basedir=[path_dir_workdir]

2) Train models:
- python ./Code/scripts_experiments/train_model.py --basedir=[path_dir_workdir] --modelsdir=<dir_output_models> [IF RESTART: --in_config_file=<file_config> --is_restart_model=True]

# Predictions / Postprocessing
1) Compute predictions form trained model:
- python ./Code/scripts_experiments/predict_model.py <file_trained_model> <dir_output_predictions> --basedir=[path_dir_workdir] --in_config_file=<file_config>

2) Compute full-size probability maps from predictions:
- python ./Code/scripts_evalresults/postprocess_predictions.py <dir_output_predictions> <dir_output_probmaps> --basedir=[path_dir_workdir]

3) Compute airway binary mask from probability maps:
- python ./Code/scripts_evalresults/process_predicted_airway_tree.py <dir_output_probmaps> <dir_output_binmasks> --basedir=[path_dir_workdir]
- rm -r <dir_output_predictions>

4) [IF NEEDED] Compute largest connected component of airway binary masks:
- python ./Code/scripts_util/apply_operation_images.py <dir_output_binmasks> <dir_output_conn_binmasks> --type=firstconreg
- rm -r <dir_output_binmasks> && mv <dir_output_conn_binmasks> <dir_output_binmasks>

5) [IF NEEDED] Compute centrelines from airway binary masks:
- python ./Code/scripts_util/apply_operation_images.py <dir_output_binmasks> <dir_output_centrelines> --type=thinning

6) Compute results metrics / accuracy:
- python ./Code/scripts_evalresults/compute_result_metrics.py <dir_output_binmasks> --basedir=[path_dir_workdir] [IF NEEDED:--inputcentrelinesdir=<dir_output_centrelines>]

[ALTERNATIVE] Do steps 1-6 at once:
- python ./Code/scripts_evalresults/launch_predictions_full.py <file_trained_model> <dir_output_predictions> --basedir=[path_dir_workdir]

