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
from CommonUtil.FrocUtil import computeFROC, plotFROC, computeROC_Completeness_VolumeLeakage
from CommonUtil.FunctionsUtil import *
from CommonUtil.PlotsManager import *
from CommonUtil.WorkDirsManager import *
from Preprocessing.OperationsImages import *
import argparse
np.random.seed(2017)



def main(args):

    # ---------- SETTINGS ----------
    nameInputMasksRelPath  = 'ProcMasks'
    nameCentrelinesRelPath = 'ProcAllMasks_3b_i5'

    # Get the file list:
    namePredictionsFiles = 'predict_probmaps*.nii.gz'
    nameInputMasksFiles  = '*outerwall.nii.gz'
    nameCentrelinesFiles = '*centrelines.nii.gz'

    # template search files
    tempSearchInputFiles = 'av[0-9]*'

    # create file to save FROC values
    temp_outfilename = '%s_ROCsensTPspecFP_NEW.txt'

    # parameters
    nbr_of_thresholds = 8
    range_threshold = [-10, -3]
    #thresholds_list = (np.linspace(range_threshold[0], range_threshold[1], nbr_of_thresholds)).tolist()
    thresholds_list = (np.logspace(range_threshold[0], range_threshold[1], nbr_of_thresholds)).tolist()
    #thresholds_list += [1.0 - elem for elem in reversed(thresholds_list)]
    #nbr_of_thresholds *= 2
    allowedDistance = 0
    # ---------- SETTINGS ----------


    workDirsManager     = WorkDirsManager(args.basedir)
    BaseDataPath        = workDirsManager.getNameBaseDataPath()
    InputPredictDataPath= workDirsManager.getNameExistPath(args.basedir, args.predictionsdir)
    InputMasksPath      = workDirsManager.getNameExistPath(BaseDataPath, nameInputMasksRelPath)
    CentrelinesPath     = workDirsManager.getNameExistPath(BaseDataPath, nameCentrelinesRelPath)

    listPredictionsFiles    = findFilesDir(InputPredictDataPath, namePredictionsFiles)
    listGrndTruthMasksFiles = findFilesDir(InputMasksPath,       nameInputMasksFiles)
    listCentrelinesFiles    = findFilesDir(CentrelinesPath,      nameCentrelinesFiles)

    nbPredictionFiles = len(listPredictionsFiles)

    # Run checkers
    if (nbPredictionFiles == 0):
        message = "0 Predictions found in dir \'%s\'" %(InputPredictDataPath)
        CatchErrorException(message)


    threshold_listcases    = np.zeros((nbr_of_thresholds, nbPredictionFiles))
    sensitivity_listcases  = np.zeros((nbr_of_thresholds, nbPredictionFiles))
    FPaverage_listcases    = np.zeros((nbr_of_thresholds, nbPredictionFiles))
    completeness_listcases = np.zeros((nbr_of_thresholds, nbPredictionFiles))
    volumeleakage_listcases= np.zeros((nbr_of_thresholds, nbPredictionFiles))
    dice_coeff_listcases   = np.zeros((nbr_of_thresholds, nbPredictionFiles))

    print("IMPORTANT: List of Threshold Values: %s" % (thresholds_list))



    for i, predict_probmaps_file in enumerate(listPredictionsFiles):

        print('\'%s\'...' %(predict_probmaps_file))

        name_prefix_case = getExtractSubstringPattern(basename(predict_probmaps_file),
                                                      tempSearchInputFiles)

        for iter1_file, iter2_file in zip(listGrndTruthMasksFiles,
                                          listCentrelinesFiles):
            if name_prefix_case in iter1_file:
                grndtruth_masks_file = iter1_file
                centrelines_file     = iter2_file
                break
        #endfor
        print("assigned to '%s' and '%s'..." % (basename(grndtruth_masks_file), basename(centrelines_file)))


        predict_probmaps_array = FileReader.getImageArray(predict_probmaps_file)
        grndtruth_masks_array  = FileReader.getImageArray(grndtruth_masks_file)
        centrelines_array      = FileReader.getImageArray(centrelines_file)

        print("Predictions masks array of size: %s..." % (str(predict_probmaps_array.shape)))

        # need to convert to lists for FROC methods
        predict_probmaps_array = np.expand_dims(predict_probmaps_array, axis=0)
        grndtruth_masks_array  = np.expand_dims(grndtruth_masks_array,  axis=0)
        centrelines_array       = np.expand_dims(centrelines_array,     axis=0)


        # compute FROC: sensitivity-specificity
        print("computing FROC: sensitivity-specificity...")
        sensitivity_list, FPaverage_list = computeFROC(predict_probmaps_array,
                                                       grndtruth_masks_array,
                                                       allowedDistance,
                                                       thresholds_list)
        print("...done")


        # compute ROC: completeness-volume leakage
        print("computing ROC: completeness-volume leakage...")
        completeness_list, volumeleakage_list, dice_coeff_list = computeROC_Completeness_VolumeLeakage(predict_probmaps_array,
                                                                                                       grndtruth_masks_array,
                                                                                                       centrelines_array,
                                                                                                       thresholds_list)
        print("...done")





        out_filename = joinpathnames(InputPredictDataPath, temp_outfilename%(name_prefix_case))
        fout = open(out_filename, 'w')

        strheader = '/threshold/ /sensitivity/ /FPaverage/ /completeness/ /volume_leakage/ /dice_coeff/' +'\n'
        fout.write(strheader)

        for threshold, sensitivity, FPaverage, completeness, volumeleakage, dice_coeff in zip(thresholds_list,
                                                                                              sensitivity_list,
                                                                                              FPaverage_list,
                                                                                              completeness_list,
                                                                                              volumeleakage_list,
                                                                                              dice_coeff_list):
            strdata = str(threshold) + ' ' + str(sensitivity) + ' ' + str(FPaverage) + ' ' + str(completeness) + ' ' + str(volumeleakage) + ' ' + str(dice_coeff) + '\n'
            fout.write(strdata)
        #endfor

        fout.close()

        #store to compute average values over all cases
        threshold_listcases    [:,i] = thresholds_list
        sensitivity_listcases  [:,i] = sensitivity_list
        FPaverage_listcases    [:,i] = FPaverage_list
        completeness_listcases [:,i] = completeness_list
        volumeleakage_listcases[:,i] = volumeleakage_list
        dice_coeff_listcases   [:,i] = dice_coeff_list


        # plot FROC
        #print("ploting FROC...")
        #plotFROC(FPaverage_list, sensitivity_list)
    #endfor


    # thresholds_list   = np.mean(threshold_listcases,    axis=1)
    # sensitivity_list  = np.mean(sensitivity_listcases,  axis=1)
    # FPaverage_list    = np.mean(FPaverage_listcases,    axis=1)
    # completeness_list = np.mean(completeness_listcases, axis=1)
    # volumeleakage_list= np.mean(volumeleakage_listcases,axis=1)
    # dice_coeff_list   = np.mean(dice_coeff_listcases,   axis=1)
    #
    #
    # out_filename = joinpathnames(InputPredictDataPath, temp_outfilename %('mean'))
    # fout = open(out_filename, 'w')
    #
    # strheader = '/threshold/ /sensitivity/ /FPaverage/ /completeness/ /volume_leakage/ /dice coeff/' +'\n'
    # fout.write(strheader)
    #
    # for threshold, sensitivity, FPaverage, completeness, volumeleakage, dice_coeff in zip(thresholds_list,
    #                                                                                       sensitivity_list,
    #                                                                                       FPaverage_list,
    #                                                                                       completeness_list,
    #                                                                                       volumeleakage_list,
    #                                                                                       dice_coeff_list):
    #     strdata = str(threshold) + ' ' + str(sensitivity) + ' ' + str(FPaverage) + ' ' + str(completeness) + ' ' + str(volumeleakage) + ' ' + str(dice_coeff) + '\n'
    #     fout.write(strdata)
    # # endfor
    #
    # fout.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--predictionsdir', default='Predictions')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
