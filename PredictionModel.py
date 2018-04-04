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
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.LoadDataManager import *
from CommonUtil.WorkDirsManager import *
from Networks.Metrics import *
from Networks.Networks import *
from Postprocessing.ReconstructImages import *

TYPEDATA = 'testing'


def main():

    workDirsManager  = WorkDirsManager(BASEDIR)
    BaseDataPath     = workDirsManager.getNameDataPath(TYPEDATA)
    RawImagesDataPath= workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'RawImages')
    TestingDataPath  = workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'ProcVolsData')
    ModelsPath       = workDirsManager.getNameNewPath(BASEDIR, 'Models')
    PredictionsPath  = workDirsManager.getNameNewPath(BASEDIR, 'Predictions')

    # Get the file list:
    listTestImagesFiles = findFilesDir(TestingDataPath  + '/volsImages*.npy')
    listTestMasksFiles  = findFilesDir(TestingDataPath  + '/volsMasks*.npy' )
    listRawImagesFiles  = findFilesDir(RawImagesDataPath + '/av*.dcm')


    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)

    # Loading Model
    model = DICTAVAILNETWORKS3D[IMODEL].getModel(IMAGES_DIMS_Z_X_Y)
    model.summary()

    weightsPath = joinpathnames(ModelsPath, RESTARTFILE)
    model.load_weights(weightsPath)


    if (CROPPINGIMAGES):
        namefile_dict = joinpathnames(BaseDataPath, "boundBoxesMasks.npy")
        dict_masks_boundingBoxes = readDictionary(namefile_dict)


    for imagesFile, masksFile, rawImagesFile in zip(listTestImagesFiles, listTestMasksFiles, listRawImagesFiles):

        print('\'%s\'...' % (rawImagesFile))

        # Loading Data
        (xTest, yTest) = LoadDataManager(IMAGES_DIMS_Z_X_Y).loadData_1File_BatchGenerator(SlicingImages(IMAGES_DIMS_Z_X_Y), imagesFile, masksFile, shuffleImages=False)

        # Evaluate Model
        yPredict = model.predict(xTest, batch_size=1)

        # Compute test accuracy
        accuracy = DiceCoefficient.compute_home(yPredict, yTest)

        print('Computed accuracy: %s...' %(accuracy))


        if (RECONSTRUCTPREDICTION):
            # Reconstruct image with same dims as original DICOM

            print('Reconstructing images for predictions...')

            masks_predict_shape = FileReader.getImageSize(rawImagesFile)

            yPredict = yPredict.reshape([yPredict.shape[0], IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH])

            if (THRESHOLDOUTIMAGES):
                yPredict = getThresholdImage(yPredict, THRESHOLDVALUE)

            if (CROPPINGIMAGES):

                crop_boundingBox = dict_masks_boundingBoxes[basename(rawImagesFile)]

                masks_predict_array = ReconstructImage(IMAGES_DIMS_Z_X_Y).compute_cropped(yPredict, masks_predict_shape, crop_boundingBox)
            else:

                masks_predict_array = ReconstructImage.compute(yPredict, masks_predict_shape)


            out_namefile = joinpathnames(PredictionsPath, 'pred_acc%0.2f_' %(accuracy) + basename(rawImagesFile).replace('.dcm','.nii'))
            FileReader.writeImageArray(out_namefile, masks_predict_array)
    #endfor


if __name__ == "__main__":
    main()