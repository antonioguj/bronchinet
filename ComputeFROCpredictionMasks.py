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
from CommonUtil.ErrorMessages import *
from CommonUtil.FileReaders import *
from CommonUtil.FrocUtil import computeFROC, plotFROC
from CommonUtil.FunctionsUtil import *
from CommonUtil.PlotsManager import *
from CommonUtil.WorkDirsManager import *
from Preprocessing.OperationsImages import *
import argparse
np.random.seed(2017)



def main(args):

    # ---------- SETTINGS ----------
    nameOriginMasksRelPath = 'ProcMasks'

    # Get the file list:
    namePredictMasksFiles = '*.nii.gz'
    nameOriginMasksFiles  = '*.nii.gz'

    # template search files
    tempSearchInputFiles = 'av[0-9]*'

    # create file to save FROC values
    tempFROCvaluesFilename = '%s-FROCvalues.txt'

    # parameters
    nbr_of_thresholds = 11
    range_threshold = [0.0, 1.0]
    thresholds_list = (np.linspace(range_threshold[0], range_threshold[1], nbr_of_thresholds)).tolist()
    #thresholds_list = (np.logspace(range_threshold[0], range_threshold[1], nbr_of_thresholds)).tolist()
    allowedDistance = 0
    # ---------- SETTINGS ----------


    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameBaseDataPath()
    PredictMasksPath = workDirsManager.getNameExistPath(args.basedir, args.predictionsdir)
    OriginMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, nameOriginMasksRelPath)

    listPredictMasksFiles = findFilesDir(PredictMasksPath, namePredictMasksFiles)
    listOriginMasksFiles  = findFilesDir(OriginMasksPath,  nameOriginMasksFiles)

    nbPredictMasksFiles = len(listPredictMasksFiles)

    # Run checkers
    if (nbPredictMasksFiles == 0):
        message = "num Predictions found in dir \'%s\'" %(PredictMasksPath)
        CatchErrorException(message)

    if (args.confineMasksToLungs):

        OriginAddMasksPath = workDirsManager.getNameExistPath(BaseDataPath, 'RawAddMasks')
        nameAddMasksFiles  = '*.dcm'
        listAddMasksFiles  = findFilesDir(OriginAddMasksPath, nameAddMasksFiles)


    threshold_listcases   = np.zeros((nbr_of_thresholds, nbPredictMasksFiles))
    sensitivity_listcases = np.zeros((nbr_of_thresholds, nbPredictMasksFiles))
    FPaverage_listcases   = np.zeros((nbr_of_thresholds, nbPredictMasksFiles))


    for i, predictionsFile in enumerate(listPredictMasksFiles):

        print('\'%s\'...' %(predictionsFile))

        predict_masks_array = FileReader.getImageArray(predictionsFile)

        print("Predictions masks array of size: %s..." % (str(predict_masks_array.shape)))


        index_origin_masks = re.search(tempSearchInputFiles, predictionsFile).group(0)

        origin_masks_file = ''
        for file in listOriginMasksFiles:
            if index_origin_masks in file:
                origin_masks_file = file
                break

        print("assigned to '%s'..." % (basename(origin_masks_file)))

        origin_masks_array  = FileReader.getImageArray(origin_masks_file)

        # Turn to binary masks (0, 1)
        origin_masks_array = OperationsBinaryMasks.process_masks(origin_masks_array)


        if (args.confineMasksToLungs):
            print("Confine masks to exclude the area outside the lungs...")

            exclude_masks_file = ''
            for file in listAddMasksFiles:
                if index_origin_masks in file:
                    exclude_masks_file = file
                    break

            print("assigned to '%s'..." % (basename(exclude_masks_file)))

            exclude_masks_array = FileReader.getImageArray(exclude_masks_file)

            origin_masks_array = OperationsBinaryMasks.apply_mask_exclude_voxels_fillzero(origin_masks_array, exclude_masks_array)


        # need to convert to lists for FROC methods
        predict_masks_array = np.expand_dims(predict_masks_array, axis=0)
        origin_masks_array  = np.expand_dims(origin_masks_array,  axis=0)

        # compute FROC
        print("computing FROC...")
        print("for list of threshold values: %s" %(thresholds_list))
        sensitivity_list, FPaverage_list = computeFROC(predict_masks_array, origin_masks_array, allowedDistance, thresholds_list)
        print("...done")


        out_FROCvaluesFilename = joinpathnames(PredictMasksPath, tempFROCvaluesFilename %(filenamenoextension(origin_masks_file)))
        fout = open(out_FROCvaluesFilename, 'w')

        strheader = '/threshold/ /sensitivity/ /FPaverage/' +'\n'
        fout.write(strheader)

        for threshold, sensitivity, FPaverage in zip(thresholds_list, sensitivity_list, FPaverage_list):
            strdata = str(threshold) + ' ' + str(sensitivity) + ' ' + str(FPaverage) +'\n'
            fout.write(strdata)
        #endfor

        fout.close()

        #store to compute average values over all cases
        threshold_listcases  [:,i] = thresholds_list
        sensitivity_listcases[:,i] = sensitivity_list
        FPaverage_listcases  [:,i] = FPaverage_list


        # plot FROC
        print("plotting FROC...")
        plotFROC(FPaverage_list, sensitivity_list)
    #endfor


    thresholds_list  = np.mean(threshold_listcases,   axis=1)
    sensitivity_list = np.mean(sensitivity_listcases, axis=1)
    FPaverage_list   = np.mean(FPaverage_listcases,   axis=1)

    out_FROCvaluesFilename = joinpathnames(PredictMasksPath, tempFROCvaluesFilename %('mean'))
    fout = open(out_FROCvaluesFilename, 'w')

    strheader = '/threshold/ /sensitivity/ /FPaverage/' + '\n'
    fout.write(strheader)

    for threshold, sensitivity, FPaverage in zip(thresholds_list, sensitivity_list, FPaverage_list):
        strdata = str(threshold) + ' ' + str(sensitivity) + ' ' + str(FPaverage) + '\n'
        fout.write(strdata)
    # endfor

    fout.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--predictionsdir', default='Predictions')
    parser.add_argument('--confineMasksToLungs', type=str2bool, default=CONFINEMASKSTOLUNGS)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
