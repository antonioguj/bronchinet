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
SCRIPT_POSTPROCESSPREDICTIONS      = joinpathnames(CODEDIR, 'Scripts_Experiments/PostprocessPredictions.py')
SCRIPT_EXTRACTCENTRELINESFROMMASKS = joinpathnames(CODEDIR, 'Scripts_Util/ApplyOperationImages.py')
SCRIPT_COMPUTERESULTMETRICS        = joinpathnames(CODEDIR, 'Scripts_Experiments/ComputeResultMetrics.py')


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
    nameTempoPosteriorsRelPath   = 'PosteriorsWorkData/'
    namePosteriorsRelPath        = 'Posteriors/'
    namePredictionsRelPath       = 'BinaryMasks/'
    namePredictCentrelinesRelPath= 'Centrelines/'
    nameReferKeysPredictionsFile = 'referenceKeys_posteriors.npy'
    nameOutputResultsMetricsFile = 'result_metrics.txt'

    listResultsMetrics = ['DiceCoefficient', 'AirwayVolumeLeakage']
    # ---------- SETTINGS ----------


    inputdir = dirnamepathfile(args.inputmodelfile)
    in_cfgparams_file = joinpathnames(inputdir, NAME_CONFIGPARAMS_FILE)

    if not isExistfile(in_cfgparams_file):
        message = "Config params file not found: \'%s\'..." % (in_cfgparams_file)
        CatchErrorException(message)
    else:
        input_args_file = readDictionary_configParams(in_cfgparams_file)
    #print("Retrieve BaseDir from file: \'%s\'...\n" % (in_cfgparams_file))
    #BaseDir = str(input_args_file['basedir'])
    BaseDir = currentdir()


    # OutputBaseDir = makeUpdatedir(args.outputbasedir)
    OutputBaseDir = args.outputbasedir
    makedir(OutputBaseDir)

    InOutTempoPosteriorsPath     = joinpathnames(OutputBaseDir, nameTempoPosteriorsRelPath)
    InOutPosteriorsPath          = joinpathnames(OutputBaseDir, namePosteriorsRelPath)
    InOutPredictionsPath         = joinpathnames(OutputBaseDir, namePredictionsRelPath)
    InOutPredictCentrelinesPath  = joinpathnames(OutputBaseDir, namePredictCentrelinesRelPath)
    InOutReferKeysPosteriorsFile = joinpathnames(OutputBaseDir, nameReferKeysPredictionsFile)


    list_calls_all = []


    # 1st: Compute model predictions, and posteriors for testing work data
    new_call = ['python', SCRIPT_PREDICTIONMODEL, args.inputmodelfile,
                '--basedir', BaseDir,
                '--nameOutputPredictionsRelPath', InOutTempoPosteriorsPath,
                '--nameOutputReferKeysFile', InOutReferKeysPosteriorsFile,
                '--cfgfromfile', in_cfgparams_file,
                '--testdatadir', args.testdatadir,
                '--typeGPUinstalled', TYPEGPUINSTALLED]
    list_calls_all.append(new_call)


    # 2nd: Compure reconstructed posteriors and binary masks from posteriors work data
    new_call = ['python', SCRIPT_POSTPROCESSPREDICTIONS,
                '--basedir', BaseDir,
                '--threshold_values', ' '.join([str(el) for el in args.thresholds]),
                '--attachCoarseAirwaysMask', 'True',
                '--nameInputPredictionsRelPath', InOutTempoPosteriorsPath,
                '--nameInputReferKeysFile', InOutReferKeysPosteriorsFile,
                '--nameOutputPosteriorsRelPath', InOutPosteriorsPath,
                '--nameOutputPredictMasksRelPath', InOutPredictionsPath,
                '--masksToRegionInterest', str(MASKTOREGIONINTEREST),
                '--rescaleImages', str(RESCALEIMAGES),
                '--cropImages', str(CROPIMAGES)]
    list_calls_all.append(new_call)


    # 3rd: Compute centrelines by thinning the binary masks
    new_call = ['python', SCRIPT_EXTRACTCENTRELINESFROMMASKS, InOutPredictionsPath, InOutPredictCentrelinesPath,
                '--type', 'thinning']
    list_calls_all.append(new_call)


    # 4th: Compute testing metrics from predicted binary masks and centrelines
    new_call = ['python', SCRIPT_COMPUTERESULTMETRICS, InOutPredictionsPath,
                '--basedir', BaseDir,
                '--inputcentrelinesdir', InOutPredictCentrelinesPath,
                '--outputresultsfile', nameOutputResultsMetricsFile,
                '--removeTracheaCalcMetrics', str(REMOVETRACHEACALCMETRICS)]
    list_calls_all.append(new_call)


    # remove temporary data for posteriors not needed
    new_call = ['rm', '-r', InOutTempoPosteriorsPath]
    list_calls_all.append(new_call)
    new_call = ['rm', InOutReferKeysPosteriorsFile, InOutReferKeysPosteriorsFile.replace('.npy', '.csv')]
    list_calls_all.append(new_call)


    # move results file one basedir down
    in_resfile  = joinpathnames(InOutPredictionsPath, nameOutputResultsMetricsFile)
    out_resfile = joinpathnames(OutputBaseDir, nameOutputResultsMetricsFile)

    new_call = ['mv', in_resfile, out_resfile]
    list_calls_all.append(new_call)



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
    parser.add_argument('inputmodelfile', type=str)
    parser.add_argument('outputbasedir', type=str)
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--thresholds', type=str, nargs='*', default=[0.5])
    parser.add_argument('--testdatadir', type=str, default='TestingData/')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
