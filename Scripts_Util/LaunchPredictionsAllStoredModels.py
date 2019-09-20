#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
#######################################################################################

from Common.Constants import *
from Common.FunctionsUtil import *
from Common.WorkDirsManager import *
import subprocess
import argparse

CODEDIR                   = joinpathnames(BASEDIR, 'Code')
SCRIPTS_LAUNCHPREDICTIONS = joinpathnames(CODEDIR, 'Scripts_ResultMeasurements/LaunchPredictionsComplete.py')



def main(args):
    # ---------- SETTINGS ----------
    nameOutputDirRelPath  = joinpathnames(args.outputbasedir, 'Predictions_e%02d')
    nameInputModelFiles   = 'model_e*.pt'
    # ---------- SETTINGS ----------


    workDirsManager     = WorkDirsManager(args.basedir)
    InputModelsPath     = workDirsManager.getNameExistPath(args.inputmodeldir)
    listInputModelFiles = findFilesDirAndCheck(InputModelsPath, nameInputModelFiles)
    listInputModelFiles = sorted(listInputModelFiles, key=getIntegerInString)[264:]



    for i, in_model_file in enumerate(listInputModelFiles):
        print("\nINPUT MODEL: \'%s\'..." % (basename(in_model_file)))
        print("----------------------------------\n")

        iepoch = getIntegerInString(in_model_file)
        OutputPath = workDirsManager.getNameNewPath(nameOutputDirRelPath %(iepoch))

        Popen_obj = subprocess.Popen(['python', SCRIPTS_LAUNCHPREDICTIONS, in_model_file, OutputPath,
                                      '--testdatadir', 'Valid-CV01'])
        Popen_obj.wait()
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputmodeldir', type=str)
    parser.add_argument('outputbasedir', type=str)
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
