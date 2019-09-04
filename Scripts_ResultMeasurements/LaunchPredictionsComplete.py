#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
#######################################################################################

#!/usr/bin/python

from Common.Constants import *
from Common.FunctionsUtil import *
from Common.WorkDirsManager import *
import subprocess
import argparse

CODEDIR                = joinpathnames(BASEDIR, 'Code')
SCRIPTS_EXPERIMENTS    = joinpathnames(CODEDIR, 'Scripts_Experiments')
SCRIPTS_PREPROCESSING  = joinpathnames(CODEDIR, 'Scripts_PreprocessData')
SCRIPTS_POSTPROCESSING = joinpathnames(CODEDIR, 'Scripts_ResultMeasurements')



def main(args):
    # ---------- SETTINGS ----------
    nameOutputPosteriorsRelPath         = joinpathnames(args.outputbasedir, 'Posteriors')
    nameOutputPredictionsRelPath        = joinpathnames(args.outputbasedir, 'Predictions_Thres%s')
    nameOutputPredictCentrelinesRelPath = joinpathnames(args.outputbasedir, 'PredictCentrelines_Thres%s')

    inputdir = dirnamepathfile(args.inputmodel)
    in_cfgparams_file = joinpathnames(inputdir, 'cfgparams.txt')

    script_predictionModel             = joinpathnames(SCRIPTS_EXPERIMENTS, 'PredictionModel.py')
    script_postprocessPredictions      = joinpathnames(SCRIPTS_POSTPROCESSING, 'PostprocessPredictions.py')
    script_extractCentrelinesFromMasks = joinpathnames(SCRIPTS_PREPROCESSING, 'ExtractCentrelinesFromMasks.py')
    script_computeResultMetrics        = joinpathnames(SCRIPTS_POSTPROCESSING, 'ComputeResultMetrics.py')
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    OutputPosteriorsPath = workDirsManager.getNameNewPath(nameOutputPosteriorsRelPath)


    # 1st script: 'PredictionModel.py'
    Popen_obj = subprocess.Popen(['python', script_predictionModel, args.inputmodel, OutputPosteriorsPath])
    Popen_obj.wait()

    for i, ithres in enumerate(args.thresholds):
        OutputPredictionsPath = workDirsManager.getNameNewPath( nameOutputPredictionsRelPath %(ithres))
        OutputPredictCentrelinesPath = workDirsManager.getNameNewPath( nameOutputPredictCentrelinesRelPath %(ithres))

        # 2nd script: 'PostprocessPredictions.py'
        Popen_obj = subprocess.Popen(['python', script_postprocessPredictions, OutputPosteriorsPath, OutputPredictionsPath,
                                      '--threshold', str(ithres)])
        Popen_obj.wait()

        # 3rd script: 'ExtractCentrelinesFromMasks.py'
        Popen_obj = subprocess.Popen(['python', script_extractCentrelinesFromMasks, OutputPredictionsPath, OutputPredictCentrelinesPath])
        Popen_obj.wait()

        # 4th script: 'ComputeResultMetrics.py'
        Popen_obj = subprocess.Popen(['python', script_computeResultMetrics, OutputPredictionsPath,
                                      '--inputcentrelinesdir', OutputPredictCentrelinesPath])
        Popen_obj.wait()
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputmodel', type=str)
    parser.add_argument('outputbasedir', type=str)
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--thresholds', type=str, nargs='*', default=[0.5])
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
