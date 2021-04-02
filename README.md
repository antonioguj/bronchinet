BronchiNet
==============================

Airway segmentation from chest CTs using deep Convolutional Neural Networks

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── bronchinet         <- Source code for use in this project.
    │   │
    │   ├── common         <- General files and utilities
    │   ├── dataloaders    <- Modules to load data and batch generators
    │   ├── imageoperators <- Various image operations
    │   ├── models 	   <- All modules to define networks, metrics and optimizers
    │   ├── plotting       <- Various plotting modules
    │   ├── postprocessing <- Modules to postprocess the output of networks
    │   ├── preprocessing  <- Modules to preprocess the images to feed to networks
    │   │
    │   ├── scripts_evalresults	<- Scripts to evaluate results from models
    │   ├── scripts_experiments	<- Scripts to train and test models
    │   ├── scripts_preparedata	<- Scripts to prepare data to train models
    │   └── scripts_util	<- Scripts for various utilities
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Requirements
------------

(Recommended to use python virtualenv)
- pip install -r requirements.txt

- cuda >= 10.2 (https://developer.nvidia.com/cuda-zone)
- cuDNN >= 7.6.5 (https://developer.nvidia.com/rdp/cudnn-download)

Instructions
------------

Create working directory
------------

- mkdir <working_dir> && cd <working_dir>
- ln -s <dir_data_stored> BaseData
- ln -s <dir_this_framework> Code

(Default settings for all scripts below are in file ./bronchinet/common/constants.py)

[IF NEEDED] (in "~/.bashrc" file: export PYTHONPATH=<dir_this_framework>/bronchinet/")

Prepare data
------------

1) [IF NEEDED] Preprocess data: apply various operations to input images / masks: rescaling, binarise masks
- python ./Code/scripts_util/apply_operation_images.py <dir_input_data> <dir_output_data> --type=[various option]

2) Compute bounding-boxes around lung masks, for input images:
- python ./Code/scripts_preparedata/compute_boundingbox_images.py --datadir=[path_dir_dataset]

3) Prepare data: include i) crop images, ii) mask ground-truth to lung regions, iii) rescale images.
- python ./Code/scripts_preparedata/prepare_data.py --datadir=[path_dir_dataset]

Train models
------------

1) Distribute data in training / validation / testing:
- python ./Code/scripts_experiments/distribute_data.py --basedir=[path_dir_workdir]

2) Train models:
- python ./Code/scripts_experiments/train_model.py --basedir=[path_dir_workdir] --modelsdir=<dir_output_models> [IF RESTART: --in_config_file=<file_config> --is_restart_model=True]

Test models
------------

1) Compute predictions form trained model:
- python ./Code/scripts_experiments/predict_model.py <file_trained_model> <dir_output_predictions> --basedir=[path_dir_workdir] --in_config_file=<file_config>

2) Compute full-size probability maps from predictions:
- python ./Code/scripts_evalresults/postprocess_predictions.py <dir_output_predictions> <dir_output_probmaps> --basedir=[path_dir_workdir]

3) Compute airway binary mask from probability maps:
- python ./Code/scripts_evalresults/process_predicted_airway_tree.py <dir_output_probmaps> <dir_output_binmasks> --basedir=[path_dir_workdir]
- rm -r <dir_output_predictions>

4) Compute largest connected component of airway binary masks:
- python ./Code/scripts_util/apply_operation_images.py <dir_output_binmasks> <dir_output_conn_binmasks> --type=firstconreg
- rm -r <dir_output_binmasks> && mv <dir_output_conn_binmasks> <dir_output_binmasks>

5) Compute centrelines from airway binary masks:
- python ./Code/scripts_util/apply_operation_images.py <dir_output_binmasks> <dir_output_centrelines> --type=thinning

6) Compute results metrics / accuracy:
- python ./Code/scripts_evalresults/compute_result_metrics.py <dir_output_binmasks> <dir_output_centrelines> --basedir=[path_dir_workdir]

[ALTERNATIVE] Do steps 1-6 at once:
- python ./Code/scripts_evalresults/launch_predictions_full.py <file_trained_model> <dir_output_predictions> --basedir=[path_dir_workdir]
