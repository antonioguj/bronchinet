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
from CommonUtil.FileDataManager import *
from CommonUtil.PlotsManager import *
from CommonUtil.WorkDirsManager import *
from Networks.Metrics import *
from Networks.Networks import *
import numpy as np

BATCH_SIZE = 1

FORMATOUTIMAGES = 'nifti'

IMODEL = 'Unet3D'

#MAIN
workDirsManager = WorkDirsManager(BASEDIR)
RawMasksDataPath= workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'RawMasks')
TestingDataPath = workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'ProcVolsData')
PredictMasksPath= workDirsManager.getNameNewPath(BASEDIR, 'Predictions')
ModelsPath      = workDirsManager.getNameModelsPath()


print('-' * 30)
print('Loading saved weights...')
print('-' * 30)

model = DICTAVAILNETWORKS3D[IMODEL].getModel(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH)
model.summary()

weightsPath = ModelsPath + '/bestModel.hdf5'
model.load_weights(weightsPath)


print('-' * 30)
print('Loading data...')
print('-' * 30)

listTestImagesFiles = sorted(glob(TestingDataPath + '/volsImages*.npy'))
listTestMasksFiles  = sorted(glob(TestingDataPath + '/volsMasks*.npy' ))
listMasksDICOMfiles = sorted(glob(RawMasksDataPath + '/av*.dcm'))

for imagesFile, masksFile, masksDICOMfile in zip(listTestImagesFiles, listTestMasksFiles, listMasksDICOMfiles):

    print('\'%s\'...' % (imagesFile))

    # LOADING DATA
    (xTest, yTest) = FileDataManager.loadDataFiles3D(imagesFile, masksFile)

    masksDICOM_truth = DICOMreader.getImageArray(masksDICOMfile)

    # EVALUATE MODEL
    yPredict = model.predict(xTest, batch_size=1)
    yPredict = yPredict.reshape([yPredict.shape[0]*IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH])


    # SAVE IMAGE: same dims than original DICOM
    masks_predict = np.ndarray(masksDICOM_truth.shape, dtype=FORMATMASKDATA)
    masks_predict[:,:,:] = 0

    masks_predict[0:yPredict.shape[0],
                  CROPPATCH[0][0]:CROPPATCH[0][1],
                  CROPPATCH[1][0]:CROPPATCH[1][1]] = np.round(yPredict)

    # Invert images: input images were inverted to start from the traquea
    masks_predict = masks_predict[::-1]


    # Compute test accuracy
    accuracy = DiceCoefficient.compute_home(yPredict, yTest)

    if( FORMATOUTIMAGES=='numpy' ):

        out_namefile = os.path.join(PredictMasksPath, 'pred_acc%0.2f_'%(accuracy) + os.path.basename(masksDICOMfile).replace('.dcm','.npy'))
        NUMPYreader.writeImageArray(out_namefile, masks_predict)

    elif( FORMATOUTIMAGES=='nifti' ):

        out_namefile = os.path.join(PredictMasksPath, 'pred_acc%0.2f_'%(accuracy) + os.path.basename(masksDICOMfile).replace('.dcm','.nii'))
        NIFTIreader.writeImageArray(out_namefile, np.swapaxes(masks_predict, 0, 2))

    elif( FORMATOUTIMAGES=='dicom' ):

        out_namefile = os.path.join(PredictMasksPath, 'pred_acc%0.2f_'%(accuracy) + os.path.basename(masksDICOMfile))
        DICOMreader.writeImageArray(out_namefile, np.uint16(masks_predict))
        # copy DICOM header info: does not work
        #ds1 = dicom.read_file(masksDICOMfile)
        #ds2 = dicom.read_file(out_namefile)
        #ds1.PixelData = ds2.PixelData
        #ds1.save_as(out_namefile)
#endfor