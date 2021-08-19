MRI_enhance (Derived from BronchiNet)
================================

Enhancement method of knee MRI images using deep Convolutional Neural Networks

Contact: Antonio Garcia-Uceda Juarez (antonio.garciauceda89@gmail.com)

Introduction
------------

This software provides functionality to enhance MRI images from Gradient-Echo and Spin-Echo (3D-GRASE) MRI sequence, to resemble Fast Spin-Echo (3D-FSE) modality. The method is based on deep CNN models, and in particular the U-Net. The implementation of the segmentation method is described in:

[PUT PAPER REFERENCE WHEN SUBMITTED]

If using this software influences positively your project, please cite the above paper.

This software includes tools to i) prepare the MRI data to use with DL models, ii) perform DL experiments for training and testing. The tools are entirely implemented in Python, and models are developed in Keras/tensorflow.

Project Organization
------------

    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make train`
    ├── README.md               <- The top-level README for developers using this project
    │
    ├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                         	   generated with `pip freeze > requirements.txt`
    │
    ├── scripts_launch          <- Scripts with pipelines and PBS scripts to run in clusters
    │
    ├── setup.py                <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                     <- Source code for use in this project.
    │   │
    │   ├── common              <- General files and utilities
    │   ├── dataloaders         <- Modules to load data and batch generators
    │   ├── imageoperators      <- Various image operations
    │   ├── models              <- All modules to define networks, metrics and optimizers
    │   ├── plotting            <- Various plotting modules
    │   ├── postprocessing      <- Modules to postprocess the output of networks
    │   ├── preprocessing       <- Modules to preprocess the images to feed to networks
    │   │
    │   ├── scripts_evalresults <- Scripts to evaluate results from models
    │   ├── scripts_experiments <- Scripts to train and test models
    │   ├── scripts_preparedata <- Scripts to prepare data to train models
    │   └── scripts_util        <- Scripts for various utilities
    │
    ├── tests                   <- Tests to validate the method implementation (to be run locally)
    └── tox.ini                 <- tox file with settings for running tox; see tox.readthedocs.io

------------

<p><small>Project structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Requirements
------------

- python packages required are in "requirements.txt"
- cuda >= 10.2 (https://developer.nvidia.com/cuda-zone)
- cuDNN >= 7.6.5 (https://developer.nvidia.com/rdp/cudnn-download)

(Recommended to use python virtualenv)

- python -m venv <path_new_pyvenv>
- source <path_new_pyvenv>/bin/activate
- pip install -r requirements.txt

# Instructions
------------

## Prepare Data Directory

Before running the scripts, the user needs to prepare the data directory with the following structure:

    ├── Images     <- Store input MRI scans (in dicom or nifti format)
    └── Labels     <- Store ground-truth MRI scans (in dicom or nifti format)

## Prepare Working Directory

The user needs to prepare the working directory in the desired location, as follows:

1. mkdir <path_your_work_dir> && cd <path_your_work_dir>
2. ln -s <path_your_data_dir> BaseData
3. ln -s <path_this_repo> Code

## Run the Scripts

The user needs only to run the scripts in the directories: "scripts_evalresults", "scripts_experiments", "scripts_launch", "scripts_preparedata", "scripts_util". Each script performs a separate and well-defined operation, either to i) prepare data, ii) run experiments, or iii) evaluate results.

The scripts are called in the command line as follows:

- python <path_script> <mandat_arg1> <mandat_arg2> ... --<option_arg1>=<value> --<option_arg2>=<value> ...

  - <mandat_argN>: obligatory arguments (if any), typically for the paths of directories of input / output files
    
  - <option_argN>: optional arguments, typical for each script. The list of optional arguments for an script can be displayed by:

    - python <path_script> --help

  - For optional arguments not indicated in the command line, they take the default values in the source file: "<path_this_repo>/src/common/constant.py"

(IMPORTANT): set the variable PYTHONPATH with the path of this code as follows:

- export PYTHONPATH=<path_this_repo>/src/

## Important Scripts 

### Steps to Prepare Data

1\. From the data directory above, create the working data used for training / testing:

- python <path_this_repo>/src/scripts_preparedata/prepare_data.py --datadir=<path_data_dir>

Several preprocessing operations can be applied in this script:

1. mask ground-truth to ROI: knees

### Steps to Train Models

1\. Distribute the working data in training / validation / testing:

- python <path_this_repo>/src/scripts_experiments/distribute_data.py --basedir=<path_work_dir>

2\. Launch a training experiment:

- python <path_this_repo>/src/scripts_experiments/train_model.py --basedir=<path_work_dir> --modelsdir=<path_output_models> 

OR restart a previous training experiment:

- python <path_this_repo>/src/scripts_experiments/train_model.py --basedir=<path_work_dir> --modelsdir=<path_stored_models> --is_restart=True --in_config_file=<path_config_file>

### Steps to Test Models

1\. Compute predictions from a trained model:

- python <path_this_repo>/src/scripts_experiments/predict_model.py <path_trained_model> <path_output_temp_predictions> --basedir=<path_work_dir> --in_config_file=<path_config_file>)

The output predictions have the format and dimensions as the working data used for testing, which is typically different from that of the original data (if using options above for preprocessing in the script "prepare_data.py").

2\. Compute predictions in format and dimensions of original data:

- python <path_this_repo>/src/scripts_evalresults/postprocess_predictions.py <path_output_temp_predictions> <path_output_predictions> --basedir=<path_work_dir>
- rm -r <path_output_temp_predictions>

3\. Compute the desired metrics from the results:

- python <path_this_repo>/src/scripts_evalresults/compute_result_metrics.py <path_output_predictions> --basedir=<path_work_dir>

### Other Scripts

The user can apply various operations to input images / masks, such as i) binarise masks, ii) mask images to a mask, iii) rescale images... as follows:

- python <path_this_repo>/src/scripts_util/apply_operation_images.py <path_input_files> <path_output_files> --type=<various_options>

Some operations require extra input arguments. To visualize the list of operations available and the required input arguments, include "--help" after the script.
