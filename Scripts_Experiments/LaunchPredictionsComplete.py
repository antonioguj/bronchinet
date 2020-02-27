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

CODEDIR                            = joinpathnames(BASEDIR, 'Code/')
SCRIPT_PREDICTIONMODEL             = joinpathnames(CODEDIR, 'Scripts_Experiments/PredictionModel.py')
SCRIPT_POSTPROCESSPREDICTIONS      = joinpathnames(CODEDIR, 'Scripts_ImageOperations/PostprocessPredictions.py')
SCRIPT_EXTRACTCENTRELINESFROMMASKS = joinpathnames(CODEDIR, 'Scripts_ImageOperations/ApplyOperationImages.py')
SCRIPT_COMPUTERESULTMETRICS        = joinpathnames(CODEDIR, 'Scripts_ImageOperations/ComputeResultMetrics.py')


def printCall(new_call):
    message = ' '.join(new_call)
    print("*" * 100)
    print("<<< Launch: %s >>>" %(message))
    print("*" * 100 +"\n")

def launchCall(new_call):
    Popen_obj = subprocess.Popen(new_call)
    Popen_obj.wait()



def main(args):
    # ---------- SETTINGS ----------
    nameOutputPosteriorsRelPath         = joinpathnames(args.outputbasedir, 'Posteriors')
    nameOutputPredictionsRelPath        = joinpathnames(args.outputbasedir, 'Predictions_Thres%1.2f')
    nameOutputPredictCentrelinesRelPath = joinpathnames(args.outputbasedir, 'PredictCentrelines_Thres%1.2f')

    inputdir = dirnamepathfile(args.inputmodel)
    in_cfgparams_file = joinpathnames(inputdir, NAME_CONFIGPARAMS_FILE)
    # ---------- SETTINGS ----------


    workDirsManager      = WorkDirsManager(args.basedir)
    OutputPosteriorsPath = workDirsManager.getNameNewPath(nameOutputPosteriorsRelPath)


    list_calls_all = []


    # 1st script: Compute predicted posteriors probabilities from networks
    new_call = ['python', SCRIPT_PREDICTIONMODEL, args.inputmodel, OutputPosteriorsPath,
                '--cfgfromfile', in_cfgparams_file,
                '--testdatadir', args.testdatadir]
    list_calls_all.append(new_call)


    for i, ithres in enumerate(args.thresholds):
        OutputPredictionsPath        = workDirsManager.getNameNewPath( nameOutputPredictionsRelPath %(ithres))
        OutputPredictCentrelinesPath = workDirsManager.getNameNewPath( nameOutputPredictCentrelinesRelPath %(ithres))


        # 2nd script: Compute predicted binary masks by thresholding the posteriors
        new_call = ['python', SCRIPT_POSTPROCESSPREDICTIONS, OutputPosteriorsPath, OutputPredictionsPath,
                    '--threshold', str(ithres)]
        list_calls_all.append(new_call)


        # 3rd script: Compute predicted centrelines by thinning the binary masks
        new_call = ['python', SCRIPT_EXTRACTCENTRELINESFROMMASKS, OutputPredictionsPath, OutputPredictCentrelinesPath,
                    '--type', 'thinning']
        list_calls_all.append(new_call)


        # 4th script: Compute testing metrics from predicted binary masks and centrelines
        new_call = ['python', SCRIPT_COMPUTERESULTMETRICS, OutputPredictionsPath,
                    '--inputcentrelinesdir', OutputPredictCentrelinesPath]
        list_calls_all.append(new_call)


        # move final res files
        in_resfile  = joinpathnames(OutputPredictionsPath, 'result_metrics_notrachea.txt')
        out_resfile = joinpathnames(args.outputbasedir, 'result_metrics_notrachea.txt')

        new_call = ['mv', in_resfile, out_resfile]
        list_calls_all.append(new_call)
    #endfor


    # Iterate over the list and carry out call serially
    for icall in list_calls_all:
        printCall(icall)
        try:
            launchCall(icall)
        except Exception as ex:
            traceback.print_exc(file=sys.stdout)
            message = 'Call failed. Stop pipeline...'
            CatchErrorException(message)
        print('\n')
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
