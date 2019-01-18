# AirwaySegmentationFramework
Framework to do airway segmentation from chest CTs using deep Convolutional Neural Networks.
Implemented in Pytorch, using deep learning libraries i) Keras or ii) Pytorch.

# Dependencies
Need to intall the following python packages (recommended to use virtualenv):
(general)
- numpy 
- matplotlib
- h5py
(image processing)
- pydicom
- nibabel
- pillow
- scikit-image
(deep learning)
- tensorflow
- tensorflow-gpu
- scikit-learn
- keras
- torch
And the following libraries to run deep learning on GPUs:
- cuda: version 9.0. https://developer.nvidia.com/cuda-zone
- cuDNN: https://developer.nvidia.com/rdp/cudnn-download

# Using the framework
Starting from the working directory, create "BaseData" dir where to store data, and link the code dir to "Code".
All user input parameters can be set in the file ./Code/CommonUtil/Constants.py, or input arguments to the given pytho script
The paths to input / output files can be set at the beginning of the python script, in the part within "SETTINGS"

1) Create processed input data (output dirs "./BaseData/ProcImagesExperData" and "./BaseData/ProcImagesExperData"):
python ./Code/PreprocessInputData.py


2) Distribute input data to training / validation / testing (output dirs in "./TrainingData", "./ValidationData" and "./TestingData"):
python ./Code/DistributeProcessData.py

3) Train model (output dir "./Models"):
python ./Code/TrainingModel.py

4) Compute predictions from model (default output dir "./Predictions_NEW"):
python ./Code/PredictionModel.py