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
from Common.FunctionsUtil import *
import numpy as np
import argparse



def main(args):
    # ---------- SETTINGS ----------
    InputDataPath = args.inputdir
    OutputDataPath = args.outputdir
    nameOutputFiles = 'file%0.2i_%s'
    # ---------- SETTINGS ----------

    listInputFiles = findFilesDirAndCheck(InputDataPath)
    num_infiles = len(listInputFiles)



    if args.random_order:
        if args.random_order_seed is not None:
            np.random.seed(args.random_order_seed)

        list_order_indexes = np.random.choice(num_infiles, size=num_infiles, replace=False)

    else:
        if args.fromfile:
            if not isExistfile(args.filename):
                message = "File \'%s\' not found..." % (args.filename)
                CatchErrorException(message)

            fout = open(args.filename, 'r')
            list_refer_filenames = [filenamenoextension(elem) for elem in fout.readlines()]


            list_order_indexes = []
            for in_refer_filename in list_refer_filenames:
                input_match_filename = findFileWithSamePrefix(in_refer_filename, listInputFiles)
                index_input_filename = listInputFiles.index(input_match_filename)
                list_order_indexes.append(index_input_filename)
            # endfor

        else:
            list_order_indexes = range(num_infiles)


    for i, index in enumerate(list_order_indexes):
        input_filename = listInputFiles[index]
        base_output_name = nameOutputFiles %(i+1, basename(input_filename))
        output_filename = joinpathnames(OutputDataPath, base_output_name)
        print("%s --> %s" %(base_output_name, input_filename))

        movefile(input_filename, output_filename)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--datadir', type=str, default=BASEDIR)
    parser.add_argument('--random_order', type=str2bool, default=False)
    parser.add_argument('--random_order_seed', type=int, default=None)
    parser.add_argument('--fromfile', type=str2bool, default=False)
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)