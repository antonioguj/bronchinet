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

CODEDIR                            = joinpathnames(BASEDIR, 'Code')
SCRIPT_PREDICTIONMODEL             = joinpathnames(CODEDIR, 'Scripts_Experiments/PredictionModel.py')
SCRIPT_POSTPROCESSPREDICTIONS      = joinpathnames(CODEDIR, 'Scripts_ResultMeasurements/PostprocessPredictions.py')
SCRIPT_EXTRACTCENTRELINESFROMMASKS = joinpathnames(CODEDIR, 'Scripts_PreprocessData/ExtractCentrelinesFromMasks.py')
SCRIPT_COMPUTERESULTMETRICS        = joinpathnames(CODEDIR, 'Scripts_ResultMeasurements/ComputeResultMetrics.py')



def main(args):
    # ---------- SETTINGS ----------
    nameOutputPosteriorsRelPath         = joinpathnames(args.outputbasedir, 'Posteriors')
    nameOutputPredictionsRelPath        = joinpathnames(args.outputbasedir, 'Predictions_Thres%s')
    nameOutputPredictCentrelinesRelPath = joinpathnames(args.outputbasedir, 'PredictCentrelines_Thres%s')

    inputdir = dirnamepathfile(args.inputmodel)
    in_cfgparams_file = joinpathnames(inputdir, 'cfgparams.txt')
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    OutputPosteriorsPath = workDirsManager.getNameNewPath(nameOutputPosteriorsRelPath)


    # 1st script: 'PredictionModel.py'
    Popen_obj = subprocess.Popen(['python', SCRIPT_PREDICTIONMODEL, args.inputmodel, OutputPosteriorsPath,
                                  '--cfgfromfile', in_cfgparams_file, '--testdatadir', args.testdatadir])
    Popen_obj.wait()

    for i, ithres in enumerate(args.thresholds):
        OutputPredictionsPath = workDirsManager.getNameNewPath( nameOutputPredictionsRelPath %(ithres))
        OutputPredictCentrelinesPath = workDirsManager.getNameNewPath( nameOutputPredictCentrelinesRelPath %(ithres))

        # 2nd script: 'PostprocessPredictions.py'
        Popen_obj = subprocess.Popen(['python', SCRIPT_POSTPROCESSPREDICTIONS, OutputPosteriorsPath, OutputPredictionsPath,
                                      '--threshold', str(ithres)])
        Popen_obj.wait()

        # 3rd script: 'ExtractCentrelinesFromMasks.py'
        Popen_obj = subprocess.Popen(['python', SCRIPT_EXTRACTCENTRELINESFROMMASKS, OutputPredictionsPath, OutputPredictCentrelinesPath])
        Popen_obj.wait()

        # 4th script: 'ComputeResultMetrics.py'
        Popen_obj = subprocess.Popen(['python', SCRIPT_COMPUTERESULTMETRICS, OutputPredictionsPath,
                                      '--inputcentrelinesdir', OutputPredictCentrelinesPath])
        Popen_obj.wait()

        # move final res file
        in_resfile  = joinpathnames(OutputPredictionsPath, 'result_metrics_notrachea.txt')
        out_resfile = joinpathnames(args.outputbasedir, 'result_metrics_notrachea.txt')

        movefile(in_resfile, out_resfile)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputmodel', type=str)
    parser.add_argument('outputbasedir', type=str)
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--thresholds', type=str, nargs='*', default=[0.5])
    parser.add_argument('--testdatadir', type=str, default='TestingData')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
