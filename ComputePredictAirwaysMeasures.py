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
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import *
from Networks_Keras.Metrics import *
from Preprocessing.OperationsImages import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    nameInputMasksRelPath  = 'Myrian-opfronted_3b_i5'
    nameCentrelinesRelPath = 'Myrian-opfronted_3b_i5'
    namePredictMasksFiles  = '*thres0-5.nii.gz'
    nameInputMasksFiles    = '*outerwall.nii.gz'
    nameCentrelinesFiles   = '*centrelines_smoothed.nii.gz'

    # template search files
    tempSearchInputFiles  = 'av[0-9]*'

    # create file to save FROC values
    temp_outfilename  = 'res_completeness_voleakage.txt'
    # ---------- SETTINGS ----------


    workDirsManager      = WorkDirsManager(args.basedir)
    BaseDataPath         = workDirsManager.getNameBaseDataPath()
    InputPredictMasksPath= workDirsManager.getNameExistPath(args.basedir, args.predictionsdir)
    InputMasksPath       = workDirsManager.getNameExistPath(BaseDataPath, nameInputMasksRelPath)
    CentrelinesPath      = workDirsManager.getNameExistPath(BaseDataPath, nameCentrelinesRelPath)

    listPredictMasksFiles    = findFilesDir(InputPredictMasksPath,namePredictMasksFiles)
    listGrndTruthMasksFiles  = findFilesDir(InputMasksPath,       nameInputMasksFiles)
    listCentrelinesFiles     = findFilesDir(CentrelinesPath,      nameCentrelinesFiles)

    nbPredictionsFiles    = len(listPredictMasksFiles)
    nbGrndTruthMasksFiles = len(listGrndTruthMasksFiles)
    nbCentrelinesFiles    = len(listCentrelinesFiles)

    # Run checkers
    if (nbPredictionsFiles == 0):
        message = "0 Predictions found in dir \'%s\'" %(InputPredictMasksPath)
        CatchErrorException(message)
    if (nbGrndTruthMasksFiles == 0):
        message = "0 Ground-truth Masks found in dir \'%s\'" %(InputMasksPath)
        CatchErrorException(message)
    if (nbGrndTruthMasksFiles != nbCentrelinesFiles):
        message = "num Ground-truth Masks %i not equal to num Centrelines %i" %(nbGrndTruthMasksFiles, nbCentrelinesFiles)
        CatchErrorException(message)


    out_filename = joinpathnames(InputPredictMasksPath, temp_outfilename)
    fout = open(out_filename, 'w')

    strheader = '/case/ /completeness/ /volume_leakage/ /dice_coeff/' +'\n'
    fout.write(strheader)


    completeness_list  = []
    volumeleakage_list = []
    dicecoeff_list     = []


    for i, predict_masks_file in enumerate(listPredictMasksFiles):

        print('\'%s\'...' %(predict_masks_file))

        name_prefix_case = getExtractSubstringPattern(basename(predict_masks_file),
                                                      tempSearchInputFiles)

        for iterfile_1, iterfile_2 in zip(listGrndTruthMasksFiles,
                                          listCentrelinesFiles):
            if name_prefix_case in iterfile_1:
                grndtruth_masks_file = iterfile_1
                centrelines_file     = iterfile_2
        #endfor
        print("assigned to '%s' and '%s'..." %(basename(grndtruth_masks_file),
                                                     basename(centrelines_file)))

        predict_masks_array   = FileReader.getImageArray(predict_masks_file)
        grndtruth_masks_array = FileReader.getImageArray(grndtruth_masks_file)
        centrelines_array     = FileReader.getImageArray(centrelines_file)


        dicecoeff = DiceCoefficient().compute_np(grndtruth_masks_array, predict_masks_array)

        completeness = AirwayCompleteness().compute_np(centrelines_array, predict_masks_array) * 100

        volumeleakage = AirwayVolumeLeakage().compute_np(grndtruth_masks_array, predict_masks_array) * 100


        completeness_list .append(completeness)
        volumeleakage_list.append(volumeleakage)
        dicecoeff_list   .append(dicecoeff)

        strdata = '\'%s\' %0.3f %0.3f %0.6f\n'%(name_prefix_case, completeness, volumeleakage, dicecoeff)
        fout.write(strdata)

        print("Computed Dice coefficient: %s..." %(dicecoeff))
        print("Computed Completeness: %s..." %(completeness))
        print("Computed Volume Leakage: %s..." % (volumeleakage))
    #endfor


    # completeness_mean = np.mean(completeness_list)
    # volumeleakage_mean= np.mean(volumeleakage_list)
    # dicecoeff_mean    = np.mean(dicecoeff_list)
    #
    # strdata = str(name_prefix_case) + ' ' + str(completeness_mean) + ' ' + str(volumeleakage_mean) + ' ' + str(dicecoeff_mean) +'\n'
    # fout.write(strdata)
    #
    # print("Mean Dice coefficient: %s..." % (dicecoeff))
    # print("Mean Completeness: %s..." % (completeness))
    # print("Mean Volume Leakage 1: %s..." % (volumeleakage))

    fout.close()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--predictionsdir', default='Predictions_NEW')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
