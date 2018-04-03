#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from CommonUtil.LoadDataManager import *
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import *
from Networks.Metrics import *
from Networks.Networks import *
from Postprocessing.ReconstructImages import *

TYPEDATA = 'testing'



#MAIN
workDirsManager  = WorkDirsManager(BASEDIR)
BaseDataPath     = workDirsManager.getNameDataPath(TYPEDATA)
RawMasksDataPath = workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'RawMasks')
TestingDataPath  = workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'ProcVolsData')
PredictionsPath  = workDirsManager.getNameNewPath(BASEDIR, 'Predictions')
ModelsPath       = workDirsManager.getNameModelsPath()


# Get the file list:
listTestImagesFiles = findFilesDir(TestingDataPath  + '/volsImages*.npy')
listTestMasksFiles  = findFilesDir(TestingDataPath  + '/volsMasks*.npy' )
listRawMasksFiles   = findFilesDir(RawMasksDataPath + '/av*.dcm')


print('-' * 30)
print('Loading saved weights...')
print('-' * 30)

# Loading Model
model = DICTAVAILNETWORKS3D[IMODEL].getModel(IMAGES_DIMS_Z_X_Y)
model.summary()

weightsPath = joinpathnames(ModelsPath, 'bestModel.hdf5')
model.load_weights(weightsPath)


if (CROPPINGIMAGES):
    namefile_dict = joinpathnames(BaseDataPath, "boundBoxesMasks.npy")
    dict_masks_boundingBoxes = readDictionary(namefile_dict)


for imagesFile, masksFile, rawMasksFile in zip(listTestImagesFiles, listTestMasksFiles, listRawMasksFiles):

    print('\'%s\'...' % (imagesFile))

    # Loading Data
    (xTest, yTest) = FileDataManager.loadDataFiles3D(imagesFile, masksFile)

    # Evaluate Model
    yPredict = model.predict(xTest, batch_size=1)

    # Compute test accuracy
    accuracy = DiceCoefficient.compute_home(yPredict, yTest)


    if (RECONSTRUCTPREDICTION):
        # Reconstruct image with same dims as original DICOM

        print('Reconstructing images for predictions...')

        masks_array = FileReader.getImageArray(rawMasksFile)

        yPredict = yPredict.reshape([yPredict.shape[0], IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH])

        if (THRESHOLDOUTIMAGES):
            yPredict = getThresholdImage(yPredict, THRESHOLDVALUE)

        if (CROPPINGIMAGES):

            crop_boundingBox = dict_masks_boundingBoxes[basename(imagesFile)]

            masks_predict_array = ReconstructImage.compute_cropped(masks_array, yPredict, crop_boundingBox)
        else:

            masks_predict_array = ReconstructImage.compute(masks_array, yPredict)

        # Revert Images to start from Traquea
            masks_predict_array = revertStackImages(masks_predict_array)


        out_namefile = joinpathnames(PredictionsPath, 'pred_acc%0.2f_' %(accuracy) + basename(rawMasksFile).replace('.dcm','.nii'))
        FileReader.writeImageArray(out_namefile, masks_predict_array)
#endfor