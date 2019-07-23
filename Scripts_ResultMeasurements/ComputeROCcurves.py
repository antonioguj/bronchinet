#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Common.WorkDirsManager import *
from DataLoaders.FileReaders import *
from PlotsManager.FrocUtil import computeFROC, computeROC_Completeness_VolumeLeakage
from Preprocessing.OperationImages import *
import argparse
np.random.seed(2017)



def main(args):
    # ---------- SETTINGS ----------
    nameInputRelPath = args.inputdir
    nameReferMasksRelPath  = 'Airways_Full'
    nameRoiMasksRelPath = 'Lungs_Full'
    nameCentrelinesRelPath = 'Centrelines_Full'
    nameOutputRelPath = args.outputdir

    nameInputFiles = '*.nii.gz'
    nameReferMasksFiles  = '*_lumen_maskedToLungs.nii.gz'
    nameRoiMasksFiles = '*_lungs.nii.gz'
    nameCentrelinesFiles = '*centrelines_maskedToLungs.nii.gz'

    # template search files
    tempSearchInputFiles = 'av[0-9]*'

    # create file to save FROC values
    temp_outfilename = 'dataFROC_UNetGNN-DynAdj_%s.txt'

    # parameters
    #um_thresholds = 9
    #range_threshold = [0.1, 0.9]
    #thresholds_list = (np.linspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()
    num_thresholds = 5
    range_threshold = [-10, -6]
    thresholds_list = (np.logspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()
    #thresholds_list = [1.0 - elem for elem in reversed(thresholds_list)]
    allowedDistance = 0
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    #BaseDataPath = workDirsManager.getNameBaseDataPath()
    BaseDataPath = '/home/antonio/Data/DLCST_Processed/'
    InputDataPath = workDirsManager.getNameExistPath(args.basedir, args.inputdir)
    ReferMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameReferMasksRelPath)
    CentrelinesPath = workDirsManager.getNameExistPath(BaseDataPath, nameCentrelinesRelPath)
    OutputDataPath = workDirsManager.getNameNewPath(args.basedir, args.outputdir)

    listInputFiles = findFilesDir(InputDataPath, nameInputFiles)
    listReferMasksFiles = findFilesDir(ReferMasksPath, nameReferMasksFiles)
    listCentrelinesFiles = findFilesDir(CentrelinesPath, nameCentrelinesFiles)

    nbInputFiles = len(listInputFiles)

    # Run checkers
    if (nbInputFiles == 0):
        message = "0 Predictions found in dir \'%s\'" %(InputDataPath)
        CatchErrorException(message)

    print("IMPORTANT: List of Threshold Values: %s" % (thresholds_list))

    threshold_listcases    = np.zeros((num_thresholds, nbInputFiles))
    completeness_listcases = np.zeros((num_thresholds, nbInputFiles))
    volumeleakage_listcases= np.zeros((num_thresholds, nbInputFiles))
    dicecoeff_listcases    = np.zeros((num_thresholds, nbInputFiles))



    for i, input_file in enumerate(listInputFiles):
        print("\nInput: \'%s\'...'" %(basename(input_file)))
        basename_file = filenamenoextension(input_file)

        for iter_file1, iter_file2 in zip(listReferMasksFiles, listCentrelinesFiles):
            if basename_file in iter_file1 and \
                basename_file in iter_file2:
                in_refermasks_file = iter_file1
                in_centrelines_file = iter_file2
                break
        #endfor
        print("Refer mask file: \'%s\'..." % (basename(in_refermasks_file)))
        print("Centrelines file: \'%s\'..." % (basename(in_centrelines_file)))

        in_probmaps_array = FileReader.getImageArray(input_file)
        refermasks_array = FileReader.getImageArray(in_refermasks_file)
        centrelines_array = FileReader.getImageArray(in_centrelines_file)
        print("Probability maps of size: %s..." % (str(in_probmaps_array.shape)))

        # need to convert to lists for FROC methods
        in_probmaps_array = np.expand_dims(in_probmaps_array, axis=0)
        refermasks_array = np.expand_dims(refermasks_array, axis=0)
        centrelines_array = np.expand_dims(centrelines_array, axis=0)


        # compute FROC: completeness-volume leakage
        print("computing FROC: completeness-volume leakage...")
        completeness_list, volumeleakage_list, dicecoeff_list = computeROC_Completeness_VolumeLeakage(in_probmaps_array,
                                                                                                      refermasks_array,
                                                                                                      centrelines_array,
                                                                                                      thresholds_list)
        print("...done")

        out_filename = joinpathnames(OutputDataPath, temp_outfilename%(basename_file))
        if isExistfile(out_filename):
            fout = open(out_filename, 'a')
        else:
            fout = open(out_filename, 'w')
            strheader = '/threshold/ /Dice/ /Completeness/ /Leakage/\n'
            fout.write(strheader)

        for threshold, completeness, volumeleakage, dicecoeff in zip(thresholds_list,
                                                                     completeness_list,
                                                                     volumeleakage_list,
                                                                     dicecoeff_list):
            strdata = '%s %s %s %s\n' %(threshold, dicecoeff, completeness, volumeleakage)
            fout.write(strdata)
        #endfor
        fout.close()

        #store to compute average values over all cases
        threshold_listcases    [:,i] = thresholds_list
        completeness_listcases [:,i] = completeness_list
        volumeleakage_listcases[:,i] = volumeleakage_list
        dicecoeff_listcases    [:,i] = dicecoeff_list
    #endfor


    thresholds_list   = np.mean(threshold_listcases,    axis=1)
    completeness_list = np.mean(completeness_listcases, axis=1)
    volumeleakage_list= np.mean(volumeleakage_listcases,axis=1)
    dicecoeff_list    = np.mean(dicecoeff_listcases,    axis=1)


    out_filename = joinpathnames(OutputDataPath, temp_outfilename%('mean'))
    if isExistfile(out_filename):
        fout = open(out_filename, 'a')
    else:
        fout = open(out_filename, 'w')
        strheader = '/threshold/ /Dice/ /Completeness/ /Leakage/\n'
        fout.write(strheader)

    for threshold, completeness, volumeleakage, dicecoeff in zip(thresholds_list,
                                                                 completeness_list,
                                                                 volumeleakage_list,
                                                                 dicecoeff_list):
        strdata = '%s %s %s %s\n' % (threshold, dicecoeff, completeness, volumeleakage)
        fout.write(strdata)
    # endfor
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--inputdir')
    parser.add_argument('--outputdir')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
