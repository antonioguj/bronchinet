#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.ErrorMessages import *
from CommonUtil.FileReaders import *
from Preprocessing.OperationsImages import *


def main(input_file, type_oper, output_file):

    #read input image
    in_image_array = FileReader.getImageArray(input_file)

    if type_oper == 'normalize':
        out_image_array = NormalizeImages.compute3D(in_image_array)
    else:
        raise Exception("ERROR. type operation '%s' not found... EXIT" %(type_oper))

    #save output image
    FileReader.writeImageArray(output_file, out_image_array)


if __name__ == "__main__":

    if( len(sys.argv)<4 ):
        print("ERROR. Number inputs not correct. Please input 'input_file', 'type_oper', 'output_file'... EXIT")
        sys.exit(0)

    input_file  = sys.argv[1]
    type_oper   = sys.argv[2]
    output_file = sys.argv[3]

    try:
        main(input_file, type_oper, output_file)
    except Exception as excp:
        print(excp.message)
        sys.exit(0)