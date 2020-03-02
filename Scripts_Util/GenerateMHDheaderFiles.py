#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from DataLoaders.FileReaders import *
from Common.FunctionsUtil import *
import numpy as np
import argparse



def main(args):

    namesOutputFiles = lambda in_name: basenameNoextension(in_name) + '.mhd'

    listInputFiles = findFilesDirAndCheck(args.inputdir)
    makedir(args.outputdir)


    for in_file in listInputFiles:
        print("\nInput: \'%s\'..." % (basename(in_file)))

        in_image_size = FileReader.getImageSize(in_file)
        out_image_size = np.roll(in_image_size, 2)

        ndims = len(out_image_size)
        str_out_image_size = ' '.join([str(elem) for elem in out_image_size])
        name_raw_file = basenameNoextension(in_file) + '.raw'


        #contruct mhd header file
        str_info  = 'ObjectType = Image\n'
        str_info += 'NDims = %s\n' %(ndims)
        str_info += 'BinaryData = True\n'
        str_info += 'DimSize = %s\n' %(str_out_image_size)
        str_info += 'ElementType = MET_UCHAR\n'
        str_info += 'ElementDataFile = %s' %(name_raw_file)

        out_file = joinpathnames(args.outputdir, namesOutputFiles(in_file))
        print("Output: \'%s\'..." % (basename(out_file)))

        fout = open(out_file, 'w')
        fout.write(str_info)
        fout.close()
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)