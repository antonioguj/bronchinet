# AirwaySegmentationFramework
Framework to do airway segmentation from chest CTs using deep Convolutional Neural Networks.
Implemented in Pytorch, using deep learning libraries i) Keras or ii) Pytorch.

# Dependencies
Need to intall the following python packages (recommended to use virtualenv):
- numpy (> 1.16.4)
- matplotlib (> 2.2.4)
- h5py (> 2.9.0)
- pydicom (> 1.0.2)
- nibabel (> 2.2.1)
- pillow (> 6.1.0)
- scikit-image (> 0.14.3)
- tensorflow (> 1.12.0)
- tensorflow-gpu (> 1.12.0)
- scikit-learn (> 0.19.1)
- keras (> 2.2.4)
- torch (> 1.0.1)

And the following libraries to run deep learning on GPUs:
- cuda: version 9.0. https://developer.nvidia.com/cuda-zone
- cuDNN: https://developer.nvidia.com/rdp/cudnn-download

# Using the framework
Starting from the working directory, create "BaseData" dir where to store data, and link the code dir to "Code".
All user input parameters can be set in the file ./Code/CommonUtil/Constants.py, or input arguments to the given pytho script
The paths to input / output files can be set at the beginning of the python script, in the part within "SETTINGS"

1) Create working directory and set-up framework:
- mkdir <working_dir> && cd <working_dir>
- ln -s <directory_where_your_data_is> BaseData
- ln -s <directory_where_you_store_this_framework> Code
- cd Code && ln -s Scripts_Experiments/* && Scripts_Preprocess/PreprocessData.py && cd -

(IF NEEDED, if Python cannot find homemade modules: export PYTHONPATH=./Code)

2) Process input data, images and labels, to format used for training algorithms (output dirs "./BaseData/Images_WorkData" and "./BaseData/Labels_WorkData"):
python ./Code/PreprocessData.py

3) Distribute input data to training / validation / testing (output dirs in "./TrainingData", "./ValidationData" and "./TestingData"):
python ./Code/DistributeProcessData.py

4) Train model (output dir "./Models"):
python ./Code/TrainingModel.py <--optional args>

5) Compute predictions from model:
python ./Code/PredictionModel.py <input_model_file '.pt'> <output_dir> <--optional args>
